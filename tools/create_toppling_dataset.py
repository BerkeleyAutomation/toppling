import argparse
import numpy as np
import os
import random
import json
from copy import deepcopy
from time import time

from autolab_core import Point, RigidTransform, YamlConfig, TensorDataset
from dexnet.envs import GraspingEnv, NoRemainingSamplesException
from dexnet.visualization import DexNetVisualizer3D as vis3d
from dexnet.constants import *
from toppling.policies import SingleTopplePolicy
from toppling import is_equivalent_pose, camera_pose, normalize, up

SEED = 107
MAX_FINAL_POSES = 20
MAX_VERTICES = 1000
CAMERA_POSE = camera_pose()
START_AT_OBJ_ID = 61

def pad_h(array, length=MAX_VERTICES):
    h, w = array.shape
    return np.vstack([array, np.zeros((length - h, w))])

def pad_w(array, width=MAX_FINAL_POSES):
    h, w = array.shape
    return np.hstack([array, np.zeros((h, width - w))])

def pad_both(array):
    return pad_w(pad_h(array))

def vis_topple_probs(vertices, topple_probs):
    vis3d.figure()
    env.render_3d_scene()
    for vertex, prob in zip(vertices, topple_probs):
       color = [min(1, 2*(1-prob)), min(2*prob, 1), 0]
       vis3d.points(Point(vertex, 'world'), scale=.001, color=color)
    for bottom_point in policy.toppling_model.bottom_points:
       vis3d.points(Point(bottom_point, 'world'), scale=.001, color=[0,0,0])
    vis3d.show(starting_camera_pose=CAMERA_POSE)

def parse_args():
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/benchmark_topple_policy_graspingenv.yaml'
    )
    parser = argparse.ArgumentParser(description='Rollout a policy for bin picking in order to evaluate performance')
    parser.add_argument('--topple_probs', action='store_false', help=
        """If specified, it will not show the topple probabilities"""
    )
    parser.add_argument('-output', type=str, help='dataset to output to')
    parser.add_argument('-num_samples', type=int, default=1000, help='how many vertices to sample')
    parser.add_argument('--config_filename', type=str, default=default_config_filename, help='configuration file to use')
    parser.add_argument('--before', action='store_true', help='Whether to show the object before the action')
    args = parser.parse_args()
    args.output = '/nfs/diskstation/db/toppling/' + args.output
    return args

def get_dataset(config, args):
    # print args
    # tensor_config = config['dataset']['tensors']
    # for field_name in ['vertices', 'normals', 'vertex_probs', 'min_required_forces']:
    #     tensor_config['fields'][field_name]['height'] = args.num_samples
    # tensor_config['fields']['final_poses']['height']

    # return TensorDataset(args.output, tensor_config)
    return TensorDataset.open(args.output, access_mode='READ_WRITE')

if __name__ ==  '__main__':
    args = parse_args()
    config = YamlConfig(args.config_filename)
    config['model']['load'] = 0
    policy = SingleTopplePolicy(config, use_sensitivity=True)

    if config['debug']:
        random.seed(SEED)
        np.random.seed(SEED)

    dataset = get_dataset(config, args)
    datapoint = dataset.datapoint_template
    
    env = GraspingEnv(config, config['vis'])
    obj_keys_to_id, obj_id = {}, START_AT_OBJ_ID
    a = time()
    # ----------------------
    # while True:
    #     try:
    #         env.reset()
    #     except NoRemainingSamplesException:
    #         break
    # ----------------------
    obj_names = []
    with open('/nfs/diskstation/db/toppling/find_objects/dexnet4.log', 'r') as file:
        for line in file:
            if line.startswith('useful'):
                obj_names.append(line.split(' ')[1][:-1])
    print obj_names
    obj_names = obj_names[START_AT_OBJ_ID:]
    print obj_names
    num_faces = []
    env.reset()
    for obj_name in obj_names:
        dataset_name, key = obj_name.split(KEY_SEP_TOKEN)
        env.state.obj = env._state_space._database.dataset(dataset_name)[key]
        # n_f = len(env.state.obj.mesh.convex_hull.faces)
        # num_faces.append(n_f)
        # continue
    # ----------------------
        print 'Computing Topples for obj', obj_id    
        env.state.material_props._color = np.array([0.5] * 3)
        policy.set_environment(env.environment)
        obj_keys_to_id[env.state.obj.key] = deepcopy(obj_id)

        obj_config = config['state_space']['object']
        stable_poses, _ = env.state.obj.mesh.compute_stable_poses(
            sigma=obj_config['stp_com_sigma'],
            n_samples=obj_config['stp_num_samples'],
            threshold=obj_config['stp_min_prob']
        )
        print 'num poses', len(stable_poses)
        for pose_num, pose in enumerate(stable_poses):
            print 'Computing Topples for pose', pose_num
            rot, trans = RigidTransform.rotation_and_translation_from_matrix(pose)
            env.state.obj.T_obj_world = RigidTransform(rot, trans, 'obj', 'world')
            # pose = env.state.obj.T_obj_world.matrix

            if args.before:
                env.render_3d_scene()        
                vis3d.show(starting_camera_pose=CAMERA_POSE)
                skip = raw_input('skip?')
                if skip == 'y':
                    continue

            vertices, normals, final_poses, vertex_probs, min_required_forces = policy.compute_topple(env.state)
            num_final_poses = len(final_poses)
            num_vertices = len(vertices)
            if num_final_poses > MAX_FINAL_POSES:
                print 'Too Many Poses!  Skipping'
                continue

            if args.topple_probs:
                topple_probs = np.sum(vertex_probs[:,1:], axis=1)
                vis_topple_probs(vertices, topple_probs)

            datapoint['obj_id'] = obj_id
            datapoint['obj_pose'] = pose
            datapoint['num_vertices'] = len(vertices)
            datapoint['vertices'] = pad_h(vertices)
            datapoint['normals'] = pad_h(normals)
            datapoint['min_required_forces'] = np.concatenate((min_required_forces, np.zeros((MAX_VERTICES - num_vertices))))
            datapoint['vertex_probs'] = pad_both(vertex_probs)
            datapoint['final_poses'] = pad_h(np.vstack([final_pose.matrix for final_pose in final_poses]), length=MAX_FINAL_POSES*4)

            dataset.add(datapoint)
        obj_id += 1
        dataset.flush()
        #break
    # print np.mean(num_faces)
    with open(args.output+"/obj_ids.json", "w") as write_file:
       json.dump(obj_keys_to_id, write_file)
    dataset.flush()
    print 'Computation time', time() - a
