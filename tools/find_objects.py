import argparse
import numpy as np
import os
import colorsys
import random
from copy import deepcopy
import logging

from autolab_core import Point, RigidTransform, YamlConfig
from dexnet.constants import *
from dexnet.envs import GraspingEnv, DexNetGreedyGraspingPolicy, NoRemainingSamplesException, NoActionFoundException
from dexnet.visualization import DexNetVisualizer3D as vis3d
from toppling.policies import SingleTopplePolicy, MultiTopplePolicy
from toppling import is_equivalent_pose, camera_pose, normalize, up
import trimesh

SEED = 107
CAMERA_POSE = camera_pose()

def to_rigid(mat):
    rot, trans = RigidTransform.rotation_and_translation_from_matrix(mat)
    return RigidTransform(rot, trans, 'obj', 'world')

def is_useful(q_value_com_pairs, key):
    for p1 in q_value_com_pairs:
        for p2 in q_value_com_pairs:
            if p1[0] > p2[0] and p1[1] < p2[1]:
                logger.info('useful '+key)
                return

def parse_args():
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/benchmark_topple_policy_graspingenv.yaml'
    )
    parser = argparse.ArgumentParser(description='Rollout a policy for bin picking in order to evaluate performance')
    parser.add_argument('--config_filename', type=str, default=default_config_filename, help='configuration file to use')
    parser.add_argument('-logfile', type=str)
    args = parser.parse_args()
    args.logfile = '/nfs/diskstation/db/toppling/find_objects/'+args.logfile
    return args

if __name__ == '__main__':
    args = parse_args()

    config = YamlConfig(args.config_filename)
    policy_params = config['policy']
    obj_config = config['state_space']['object']
    grasping_config = YamlConfig(policy_params['grasping_policy_config_filename'])
    policy = DexNetGreedyGraspingPolicy(
        grasping_config['policy']['database'], 
        grasping_config['policy']['params']
    )

    logger = logging.getLogger('toppling')
    hdlr = logging.FileHandler(args.logfile)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    env = GraspingEnv(config, config['vis'])

    while True:
        try:
            env.reset()
        except NoRemainingSamplesException:
            break
        policy.set_environment(env.environment)
        stable_poses, _ = env.state.obj.mesh.compute_stable_poses(
            sigma=obj_config['stp_com_sigma'],
            n_samples=obj_config['stp_num_samples'],
            threshold=obj_config['stp_min_prob']
        )
        q_value_com_pairs = []
        for stable_pose in stable_poses:
            env.state.obj.T_obj_world = to_rigid(stable_pose)
            com = env.state.obj.T_obj_world * Point(env.state.obj.center_of_mass, 'obj')
            try:
                action = policy.action(env.state)
                q_value = action.q_value
            except NoActionFoundException:
                q_value = 0
            q_value_com_pairs.append((q_value, com.data[2]))
        logger.info(env.state.obj.key + ' ' + str(q_value_com_pairs))
        is_useful(q_value_com_pairs, env.state.obj.key)
