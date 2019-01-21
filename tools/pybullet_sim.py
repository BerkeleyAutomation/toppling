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
    parser.add_argument('--topple_probabilities', action='store_false', help=
        """If specified, it will not show the topple probabilities"""
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
    
    observation, reward, done, info = env.step(action)
