import rospy
import logging
import argparse
import numpy as np
import os
import random
import cv2
import time
import sys, traceback
import subprocess
import json
from plyfile import PlyElement, PlyData

import autolab_core.utils as utils
from autolab_core import Point, Box, RigidTransform, YamlConfig, TensorDataset
from dexnet.constants import *
from dexnet.envs import GraspingEnv, LinearPushAction, NoRemainingSamplesException
from dexnet.visualization import DexNetVisualizer3D as vis3d
from ambidex.databases.postgres import YamlLoader, PostgresSchema
from ambidex.class_registry import postgres_base_cls_map, full_cls_list
from perception import RenderMode, PointToPlaneICPSolver, PointToPlaneFeatureMatcher, RgbdSensorFactory
from meshrender import Scene, MaterialProperties, SceneObject, VirtualCamera, SceneViewer
from toppling.policies import TestTopplePolicy
from toppling import is_equivalent_pose, camera_pose, pose_angle


SEED = 107
CAMERA_POSE = camera_pose()

class YamlObjLoader(object):
    def __init__(self, basedir):
        self.basedir = basedir
        self._map = {}
        for root, dirs, fns in os.walk(basedir):
            for fn in fns:
                full_fn = os.path.join(root, fn)
                _, f = os.path.split(full_fn)
                if f in self._map:
                    raise ValueError('Duplicate file named {}'.format(f))
                self._map[f] = full_fn
        self._yaml_loader = YamlLoader(PostgresSchema('pg_schema', postgres_base_cls_map, full_cls_list))

    def load(self, key):
        key = key + '.yaml'
        full_filepath = self._map[key]
        return self._yaml_loader.load(full_filepath)

    def clear(self):
        self._yaml_loader.clear()

    def __call__(self, key):
        return self.load(key)

class Super4PCSAligner(object):

    ply_intro = "ply\nformat binary 1.0"
    ply_vertex = "element vertex {}\nproperty float x\nproperty float y\nproperty float z\n"
    ply_outro = "end_header\n"

    def __init__(self, config):
        """Initialize a Super4PCSAligner.
        Parameters
        ----------
        config : autolab_core.YamlConfig
            Config containing information for parameterizing Super4PCS.
            Required parameters are listed in the Other Parameters section.
        Other Parameters
        ----------------
        overlap : float
            The expected overlap between future point clouds in [0, 1]
        accuracy : float
            The distance between two points for them to be considered aligned (in meters).
        samples : int
            The number of samples to take when using Super4PCS -- lower numbers are faster
            but potentially less accurate.
        timeout : int
            The number of seconds to try to align for at a maximum
        cache_dir : str
            A cache directory for the Super4PCSAligner.
        """
        self._overlap = config['overlap']
        self._accuracy = config['accuracy']
        self._samples = config['samples']
        self._timeout = config['timeout']
        self._cache_dir = config['cache_dir']

        if not os.path.exists(self._cache_dir):
            os.makedirs(self._cache_dir)
        self._points1_fn = os.path.join(self._cache_dir, 'points1.ply')
        self._points2_fn = os.path.join(self._cache_dir, 'points2.ply')
        self._tf_fn = os.path.join(self._cache_dir, 'tf.txt')
        open(self._tf_fn, 'a').close()

    def align(self, points1, points2):
        """Compute an aligning transform between two point clouds.
        Parameters
        ----------
        points1 : autolab_core.PointCloud
            The first point cloud.
        points2 : autolab_core.PointCloud
            The second point cloud.
        Returns
        -------
        autolab_core.RigidTransform
            A Rigid Tranformation taking points in the second cloud
            to the same frame as the first cloud.
        """

        # Export each point cloud as a .ply file
        data = np.zeros(len(points1.data.T), dtype=[('x','<f4'),('y','<f4'),('z','<f4')])
        data['x'] = points1.data.T[:,0]
        data['y'] = points1.data.T[:,1]
        data['z'] = points1.data.T[:,2]
        el = PlyElement.describe(data, 'vertex')
        time_start = time.time()
        PlyData([el], text=True).write(self._points1_fn)
        logging.info('Time to export {}'.format(time.time()-time_start))

        data = np.zeros(len(points2.data.T), dtype=[('x','<f4'),('y','<f4'),('z','<f4')])
        data['x'] = points2.data.T[:,0]
        data['y'] = points2.data.T[:,1]
        data['z'] = points2.data.T[:,2]
        el = PlyElement.describe(data, 'vertex')
        PlyData([el], text=True).write(self._points2_fn)
        logging.info('Time to export {}'.format(time.time()-time_start))

        # Run Super4PCS on those
        subprocess.call(["Super4PCS", "-i", self._points1_fn, self._points2_fn,
                                    "-o", str(self._overlap),
                                    "-d", str(self._accuracy),
                                    "-n", str(self._samples),
                                    "-t", str(self._timeout),
                                    "-m", self._tf_fn
                        ])


        # Read back in the transform
        mat = []
        with open(self._tf_fn) as f:
            for line in f:
                dat = line.split()
                try:
                    a = [float(dat[0]), float(dat[1]), float(dat[2]), float(dat[3])]
                    mat.append(a)
                except:
                    pass
        mat = np.array(mat)

        return RigidTransform(mat[:3,:3], mat[:3,3], from_frame=points2.frame, to_frame=points1.frame)

def create_scene(camera, workspace_objects):

    # Start with an empty scene
    scene = Scene()

    # Create a VirtualCamera
    virt_cam = VirtualCamera(camera.intrinsics, camera.pose)

    # Add the camera to the scene
    scene.camera = virt_cam
    mp = MaterialProperties(
            color=np.array([0.3,0.3,0.3]),
            k_a=0.5, k_d=0.3, k_s=0.0, alpha=10.0
    )
    if camera.geometry is not None:
        so = SceneObject(camera.geometry, camera.pose.copy(), mp)
        scene.add_object(camera.name, so)

    return scene

def get_sim_point_cloud(scene, grasp_obj):
    
    # Remove old objects
    scene_objs = scene.objects.copy()
    if 'obj' in scene_objs:
        scene.remove_object('obj')
    
    # Get graspable object material properties and add to scene
    mp = hasattr(grasp_obj, 'material_properties')
    if not mp:
        mp = MaterialProperties(
            color=np.random.uniform(0.0, 1.0, size=3),
            k_a=0.5, k_d=0.3, k_s=0.0, alpha=10.0
        )
    so = SceneObject(grasp_obj.mesh, grasp_obj.T_obj_world.copy(), mp)
    scene.add_object(grasp_obj.key, so)

    # Create simulated pointcloud for ICP matching
    wrapped_depth = depth_scene.wrapped_render([RenderMode.DEPTH])
    sim_point_cloud = phoxi_tf*phoxi.intrinsics.deproject(wrapped_depth[0])
    sim_point_cloud_masked, _ = sim_point_cloud.box_mask(mask_box)
    return sim_point_cloud_masked

def quit():
    kill_stream()
    sys.exit()

def sim_to_real_tf(sim_point_cloud_masked, vis=False):
    # Capture point cloud from physical depth camera
    _, depth_im, _ = phoxi_sensor.frames()
    phys_point_cloud = phoxi_tf*phoxi_sensor.ir_intrinsics.deproject(depth_im)
    phys_point_cloud_masked, _ = phys_point_cloud.box_mask(mask_box)
    if phys_point_cloud_masked.num_points == 0:
        logging.warn('Object not found! Skipping...')
        quit()
    
    pcs_config = {'overlap': 0.6, 'accuracy': 0.001, 'samples': 500, 'timeout': 10, 'cache_dir': '/home/chriscorrea14/Super4PCS/cache'}
    pcs_aligner = Super4PCSAligner(pcs_config)

    sim_to_real_tf = pcs_aligner.align(phys_point_cloud_masked, sim_point_cloud_masked)
    if vis:
        vis3d.figure()
        vis3d.points(phys_point_cloud_masked.data.T, color=(1,0,0), scale=.001)
        vis3d.points((sim_to_real_tf*sim_point_cloud_masked).data.T, color=(0,1,0), scale=.001)
        vis3d.show(title='Predicted pose from Sim 2 Real')
    return sim_to_real_tf

def get_dataset(config, args):
    tensor_config = config['experiments']['tensors']
    return TensorDataset(args.output, tensor_config)

def parse_args():
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/benchmark_topple_policy_graspingenv.yaml'
    )
    parser = argparse.ArgumentParser(description='Rollout a policy for bin picking in order to evaluate performance')
    parser.add_argument('--topple_probs', action='store_false', help=
        """If specified, it will not show the topple probabilities"""
    )
    parser.add_argument('-output', type=str, help='path to dataset to create')
    parser.add_argument('--config_filename', type=str, default=default_config_filename, help='configuration file to use')
    parser.add_argument('--obs', action='store_true')
    args = parser.parse_args()
    args.output = '/nfs/diskstation/db/toppling/' + args.output
    return args

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    rospy.init_node('test_toppling', anonymous=True)

    config = YamlConfig(args.config_filename)
    config['model']['load'] = 1
    policy = TestTopplePolicy(config, use_sensitivity=True)

    if config['debug']:
        random.seed(SEED)
        np.random.seed(SEED)

    # Start webcam stream and get webcam tf
    phoxi_tf = RigidTransform.load(os.path.join('/nfs/diskstation/calib', 'phoxi', 'phoxi_to_world.tf'))
    bin_tf = RigidTransform(translation=np.array([0.387500, -0.003000, -0.0025]), 
                            rotation=np.array([[-0.024541, -0.999699, 0.01000],
                                               [0.999699, -0.024541, 0.000000],
                                               [0.000000, 0.000000, 1.000000]]), 
                            from_frame='bin', to_frame='world')
    
    # Load all python objects
    basedir = os.path.join(os.path.dirname(__file__), '..', '..', 'ambidex', 'tests', 'cfg')
    yaml_obj_loader = YamlObjLoader(basedir)

    phys_robot = yaml_obj_loader('physical_yumi')
    try:
    # if True:
        work_bin = yaml_obj_loader('bin')
        work_bin.pose = bin_tf
        phoxi = yaml_obj_loader('phoxi')
        phoxi.pose = phoxi_tf
        
        # Create scene for each camera for rendering sim
        depth_scene = create_scene(phoxi, [work_bin])

        # Create ICP objects
        feature_matcher = PointToPlaneFeatureMatcher()
        solver = PointToPlaneICPSolver()

        # Start physical depth and color cameras
        logging.info('Creating Phoxi sensor')
        sensor_config = {'frame': 'phoxi', 'device_name': 1703005, 'size': 'small'}
        phoxi_sensor = RgbdSensorFactory.sensor('phoxi', sensor_config)
        logging.info('Starting Phoxi sensor')
        phoxi_sensor.start()
        logging.info('Phoxi sensor initialized')

        # Create box for filtering point cloud
        min_pt = np.array([0.25, -0.12, 0])
        max_pt = np.array([0.5, 0.12, 0.25])
        mask_box = Box(min_pt, max_pt, 'world')

        env = GraspingEnv(config, config['vis'])
        dataset = get_dataset(config, args)
        datapoint = dataset.datapoint_template

        obj_id, obj_id_to_key = 0, {}
        while not rospy.is_shutdown():
            try:
                env.reset()
            except NoRemainingSamplesException:
                break
            obj_id_to_key[obj_id] = env.state.obj.key
            print '\n\n\n\n\n\n\n\n\n\nSampled object', env.state.obj.key

            env.state.obj.T_obj_world.translation[0] += .4 # put object in bin, not outside it
            env.state.material_props._color = np.array([0.5] * 3)
            policy.set_environment(env.environment)
            
            orig_pose = env.state.T_obj_world.copy()
            sim_point_cloud_masked = get_sim_point_cloud(depth_scene, env.state.obj)
            obj_config = config['state_space']['object']
            stable_poses, _ = env.state.obj.mesh.compute_stable_poses(
                sigma=obj_config['stp_com_sigma'],
                n_samples=obj_config['stp_num_samples'],
                threshold=obj_config['stp_min_prob']
            )

            for push_num in range(10):
                push_idx = None
                # push_idx = 33
                sample_id = 0
                num_toppled, pose_angles, actual_poses = [], [], []
                print '\n\n\n\n\n\nSTARTING DIFFERENT PUSH', push_num
                while sample_id < 10:
                    sample_id += 1
                    # sim_point_cloud_masked = get_sim_point_cloud(depth_scene, env.state.obj)
                    usr_input = 'n'
                    while usr_input != 'y' and usr_input != 'a':
                        usr_input = utils.keyboard_input('\n\nPut the object in position y: continue, v: vis pose, a: abort push[y/v/a]:')
                        if usr_input == 'v':
                            vis3d.figure()
                            env.state.obj.T_obj_world = orig_pose
                            env.render_3d_scene()
                            vis3d.show(starting_camera_pose=CAMERA_POSE)
                    if usr_input == 'a':
                        print 'trying different push'
                        push_idx = None
                        sample_id = 0
                        num_toppled, pose_angles, actual_poses = [], [], []
                        continue

                    usr_input = 'n'
                    while usr_input != 'y':
                        s2r = sim_to_real_tf(sim_point_cloud_masked, vis=True)
                        usr_input = utils.keyboard_input('Did the pose registration work? [y/n]:')
                    env.state.obj.T_obj_world = s2r*orig_pose

                    # Plan topple action
                    action = policy.action(env.state, push_idx)

                    if args.topple_probs:# and push_idx == None:
                        vis3d.figure()
                        env.render_3d_scene()
                        for vertex, prob in zip(action.metadata['vertices'], action.metadata['topple_probs']):
                           color = [min(1, 2*(1-prob)), min(2*prob, 1), 0]
                           vis3d.points(Point(vertex, 'world'), scale=.001, color=color)
                        gripper = env.gripper(action)
                        # vis3d.plot3d([action.T_begin_world.translation, action.T_end_world.translation], color=(0,1,0), tube_radius=.0006)
                        pose = action.T_begin_world
                        vis3d.plot3d([pose.translation, pose.translation + .01*pose.x_axis], color=[1,0,0], tube_radius=.0006)
                        vis3d.plot3d([pose.translation, pose.translation + .01*pose.y_axis], color=[0,1,0], tube_radius=.0006)
                        vis3d.plot3d([pose.translation, pose.translation + .01*pose.z_axis], color=[0,0,1], tube_radius=.0006)
                        try:
                            vis3d.show(starting_camera_pose=CAMERA_POSE)
                        except KeyboardInterrupt:
                            print 'trying different push'
                            push_idx = None
                            sample_id = 0
                            num_toppled, pose_angles, actual_poses = [], [], []
                            continue

                    push_idx = action.metadata['best_ind']
                    final_poses = action.metadata['final_poses']
                    vertex_probs = action.metadata['vertex_probs'][push_idx]

                    # Execute action
                    phys_robot.execute(action)

                    # loops back if they always hit "n"
                    usr_input = 'n'
                    while usr_input != 'y':
                        # descending order
                        for pose_ind in np.argsort(-vertex_probs):
                            actual_pose = final_poses[pose_ind]
                            env.state.obj.T_obj_world = actual_pose
                            env.render_3d_scene()
                            vis3d.show(
                                starting_camera_pose=CAMERA_POSE, 
                                title='Is this the pose after the topple? (Pose {} {:.2f}%)'.format(pose_ind, vertex_probs[pose_ind])
                            )
                            usr_input = utils.keyboard_input('Is it in this pose? [y/n]')
                            if usr_input == 'y':
                                actual_pose = actual_pose.matrix
                                break
                        if usr_input != 'y':
                            # go through all stable poses
                            obj_config = config['state_space']['object']
                            stable_poses, _ = env.state.obj.mesh.compute_stable_poses(
                                sigma=obj_config['stp_com_sigma'],
                                n_samples=obj_config['stp_num_samples'],
                                threshold=obj_config['stp_min_prob']
                            )
                            for actual_pose in stable_poses:
                                rot, trans = RigidTransform.rotation_and_translation_from_matrix(actual_pose)
                                env.state.obj.T_obj_world = RigidTransform(rot, trans, 'obj', 'world')
                                env.render_3d_scene()
                                vis3d.show(
                                    starting_camera_pose=CAMERA_POSE, 
                                    title='Is this the pose after the topple?'.format(pose_ind)
                                )
                                usr_input = utils.keyboard_input('Is it in this pose? [y/n]')
                                if usr_input == 'y':
                                    break
                            if usr_input != 'y':
                                print 'looping back'
                    # E_pose_angle = 0
                    # for prob, final_pose in zip(vertex_probs, final_poses):
                    #     E_pose_angle += prob * pose_angle(final_pose, actual_pose)
                    # pose_angles.append(E_pose_angle)
                    num_toppled.append(1 if pose_ind != 0 else 0)
                    actual_poses.append(actual_pose)

                # saving data to datapoint
                env.state.obj.T_obj_world = orig_pose
                action = policy.action(env.state, push_idx)
                vertex = action.metadata['vertices'][push_idx]
                normal = action.metadata['normals'][push_idx]
                dataset_name = env.state.obj.key.split(KEY_SEP_TOKEN)[0]
                datapoint['obj_id'] = obj_id
                datapoint['obj_pose'] = orig_pose.matrix
                datapoint['vertex'] = vertex
                datapoint['normal'] = normal
                datapoint['fraction_toppled'] = np.mean(num_toppled)
                # datapoint['pose_angles'] = np.array(pose_angles)
                datapoint['actual_poses'] = np.vstack(actual_poses)
                print datapoint
                dataset.add(datapoint)
                dataset.flush()
                # Reset push index, try different push
                push_idx = None
            obj_id += 1
            print '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'
    except:
        traceback.print_exc(file=sys.stdout)
    dataset.flush()
    with open(args.output+"/obj_keys.json", "w") as write_file:
        json.dump(obj_id_to_key, write_file)
    phys_robot.stop()

