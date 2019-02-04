import argparse
import numpy as np
import os
import matplotlib as mpl
from copy import deepcopy
import matplotlib.pyplot as plt
import networkx as nx
import random

from autolab_core import YamlConfig
from dexnet.constants import *
from dexnet.envs import GraspingEnv
from dexnet.envs import DexNetGreedyGraspingPolicy

if __name__ == '__main__':
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/benchmark_topple_policy_graspingenv.yaml'
    )
    config = YamlConfig(default_config_filename)
    grasping_config_filename = config['policy']['grasping_policy_config_filename']
    print grasping_config_filename
    grasping_config = YamlConfig(grasping_config_filename)

    database = grasping_config['policy']['database']
    params = grasping_config['policy']['params']
    policy = DexNetGreedyGraspingPolicy(database, params)
    
    env = GraspingEnv(config, config['vis'])
    env.reset()
    policy.set_environment(env.environment)
    action = policy.action(env.state)
    print action.q_value
