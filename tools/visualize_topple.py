import argparse
import numpy as np
import os

from autolab_core import Point, RigidTransform, YamlConfig, TensorDataset
from dexnet.constants import *
from dexnet.envs import GraspingEnv
from dexnet.visualization import DexNetVisualizer3D as vis3d
from toppling.policies import TopplingPolicy

SEED = 107
CAMERA_ROT = np.array([[ 0,-1, 0],
                       [-1, 0, 0],
                       [ 0, 0,-1]])
theta = np.pi/3
c = np.cos(theta)
s = np.sin(theta)
CAMERA_ROT = np.array([[c,0,-s],
                       [0,1,0],
                       [s,0,c]]).dot(CAMERA_ROT)
theta = np.pi/6
c = np.cos(theta)
s = np.sin(theta)
#CAMERA_ROT = np.array([[c,-s,0],
#                       [s,c,0],
#                       [0,0,1]]).dot(CAMERA_ROT)
CAMERA_TRANS = np.array([-.25,-.25,.35])
CAMERA_TRANS = np.array([-.4,0,.3])
CAMERA_POSE = RigidTransform(CAMERA_ROT, CAMERA_TRANS, from_frame='camera', to_frame='world')

if __name__ == '__main__':
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/benchmark_topple_policy_graspingenv.yaml'
    )
    parser = argparse.ArgumentParser(description='Rollout a policy for bin picking in order to evaluate performance')
    parser.add_argument('--config_filename', type=str, default=default_config_filename, help='configuration file to use')
    parser.add_argument('--topple_probs', action='store_false', help=
        """If specified, it will not show the topple probabilities"""
    )
    parser.add_argument('--quality', action='store_false', help=
        """If specified, it will not show the quality increase from toppling at certain vertices"""
    )
    parser.add_argument('--topple_graph', action='store_false', help=
        """If specified, it will not show the topple graph"""
    )
    args = parser.parse_args()

    config = YamlConfig(args.config_filename)
    policy = TopplingPolicy(config['policy']['grasping_policy_config_filename'])
    
    env = GraspingEnv(config, config['vis'])
    env.reset()
    policy.set_environment(env.environment)
    action = policy.action(env.state)

    # Visualize
    if args.topple_probs:
        vis3d.figure()
        env.render_3d_scene()
        for vertex, prob in zip(action.metadata['vertices'], action.metadata['topple_probs']):
            color = np.array([min(1, 2*(1-prob)), min(2*prob, 1), 0])
            vis3d.points(Point(vertex, 'world'), scale=.0005, color=color)
        vis3d.show(starting_camera_pose=CAMERA_POSE)

    if args.quality_increases:
        vis3d.figure()
        env.render_3d_scene()
        for vertex, quality_increase in zip(action.metadata['vertices'], action.metadata['quality_increases']):
            topples = action.metadata['final_pose_ind'] != 0
            color = np.array([min(1, 2*(1-prob)), min(2*prob, 1), 0]) if topples else np.array([0,0,0])
            vis3d.points(Point(vertex, 'world'), scale=.0005, color=color)
        vis3d.show(starting_camera_pose=CAMERA_POSE)

    if args.topple_graph:
        vis3d.figure()
        env.render_3d_scene()
        final_pose_ind = action.metadata['final_pose_ind'] / float(np.amax(action.metadata['final_pose_ind']))
        for vertex, frac in zip(action.metadata['vertices'], final_pose_ind):
            color = np.array([min(1, 2*(1-frac)), 0, min(2*frac, 1)])
            vis3d.points(Point(vertex, 'world'), scale=.0005, color=color)
        vis3d.show(starting_camera_pose=CAMERA_POSE)

        pose_num = 0
        for (
            pose, 
            edge_point1, 
            edge_point2, 
            frac
        ) in zip(
            action.metadata['final_poses'], 
            action.metadata['bottom_points'], 
            np.roll(action.metadata['bottom_points'],-1,axis=0), 
            final_pose_ind
        ):
            print 'Pose:', pose_num
            pose_num += 1
            env.state.material_props._color = np.array([min(1, 2*(1-frac)), 0, min(2*frac, 1)])
            env.state.obj.T_obj_world = pose.T_obj_table
            
            vis3d.figure()
            env.render_3d_scene()
            vis3d.points(Point(edge_point1, 'world'), scale=.0005, color=np.array([0,0,1]))
            vis3d.points(Point(edge_point2, 'world'), scale=.0005, color=np.array([0,0,1]))
            vis3d.show(starting_camera_pose=CAMERA_POSE) 
