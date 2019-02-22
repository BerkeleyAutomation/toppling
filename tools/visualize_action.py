import rospy
import logging
import argparse
import numpy as np
import os
import colorsys
import random

from autolab_core import Point, RigidTransform, YamlConfig, TensorDataset
from dexnet.constants import *
from dexnet.envs import GraspingEnv
from dexnet.visualization import DexNetVisualizer3D as vis3d
from toppling.policies import SingleTopplePolicy
from toppling import is_equivalent_pose

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

def display_or_save(filename):
    if args.save:
        vis3d.save_loop(filename, starting_camera_pose=CAMERA_POSE)
    else:
        vis3d.show(starting_camera_pose=CAMERA_POSE)

def parse_args():
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/benchmark_topple_policy_graspingenv.yaml'
    )
    parser = argparse.ArgumentParser(description='Rollout a policy for bin picking in order to evaluate performance')
    parser.add_argument('--config_filename', type=str, default=default_config_filename, help='configuration file to use')
    parser.add_argument('--save', action='store_true', help='save to a picture rather than opening a window')
    parser.add_argument('--before', action='store_true', help='Whether to show the object before the action')
    return parser.parse_args()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    args = parse_args()
    print '\n\nSaving to file' if args.save else '\n\nDisplaying in a window'

    config = YamlConfig(args.config_filename)
    policy = SingleTopplePolicy(config['policy'], use_sensitivity=True)

    if config['debug']:
        random.seed(SEED)
        np.random.seed(SEED)
    
    env = GraspingEnv(config, config['vis'])
    env.reset()
    env.state.material_props._color = np.array([0.86274509803] * 3)
    obj_name = env.state.obj.key
    policy.set_environment(env.environment)
    if args.before:
        env.render_3d_scene()
        vis3d.show(starting_camera_pose=CAMERA_POSE)
    action = policy.action(env.state, env)

    vis3d.figure()
    env.render_3d_scene()
    num_vertices = len(action.metadata['vertices'])
    for i, vertex, q_increase in zip(np.arange(num_vertices), action.metadata['vertices'], action.metadata['quality_increases']):
        topples = action.metadata['final_pose_ind'][i] != 0
        color = np.array([min(1, 2*(1-q_increase)), min(2*q_increase, 1), 0]) if topples else np.array([0,0,0])
        # color = np.array([min(1, 2*(1-q_increase)), min(2*q_increase, 1), 0])
        scale = .001 if i != action.metadata['best_ind'] else .003
        vis3d.points(Point(vertex, 'world'), scale=scale, color=color)

    gripper = env.gripper(action)
    T_start_world = action.T_begin_world * gripper.T_mesh_grasp
    T_end_world = action.T_end_world * gripper.T_mesh_grasp
    vis3d.mesh(gripper.mesh, T_start_world, color=(0,1,0))
    vis3d.mesh(gripper.mesh, T_end_world, color=(1,0,0))
    display_or_save('{}_quality_increases.gif'.format(obj_name))
