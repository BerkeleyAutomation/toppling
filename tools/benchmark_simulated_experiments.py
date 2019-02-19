import argparse
import numpy as np
import os
import colorsys
import random

from autolab_core import Point, RigidTransform, YamlConfig, TensorDataset
from dexnet.constants import *
from dexnet.envs import GraspingEnv
#from dexnet.visualization import DexNetVisualizer3D as vis3d
from ambicore.visualization import Visualizer3D as vis3d
from toppling.policies import SingleTopplePolicy, MultiTopplePolicy
from toppling import is_equivalent_pose, camera_pose

SEED = 107
CAMERA_POSE = camera_pose()

def render_3d_scene(env):
    vis3d.mesh(env.state.mesh,
            env.state.T_obj_world.matrix,
            color=env.state.color)

def forward_sim(G):
    curr_node_id = 0
    curr_node = G.nodes[curr_node_id]
    print 'Original Quality', curr_node['gq']
    print 'Path: 0', 
    while True:
        best_action, best_q = None, 0
        for action_id in G.neighbors(curr_node_id):
            action = G.nodes[action_id]
            if action['value'] > best_q:
                best_q = action['value']
                best_action = action_id
        if best_q <= curr_node['gq']:
            break
        # execute best action
        next_state_edges = filter(lambda (u,a,b): u == best_action, G.edges.data())
        _, next_state_ids, probs = zip(*next_state_edges)
        probs = map(lambda prob: prob['prob'], probs)
        curr_node_id = np.random.choice(next_state_ids, p=probs)
        curr_node = G.nodes[curr_node_id]
        print curr_node_id,
    print '\nFinal Quality', curr_node['gq'], '\n'

def test_multi_push(env):
    env = GraspingEnv(config, config['vis'])
    policy = MultiTopplePolicy(config['policy'], use_sensitivity=True)
    while True:
        env.reset()
        env.state.material_props._color = np.array([0.5] * 3)
        policy.set_environment(env.environment)
        
        if args.before:
            vis3d.figure()
            render_3d_scene(env)
            vis3d.show(starting_camera_pose=CAMERA_POSE)
        action = policy.action(env.state)
        forward_sim(policy.G)

def parse_args():
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/benchmark_topple_policy_graspingenv.yaml'
    )
    parser = argparse.ArgumentParser(description='Rollout a policy for bin picking in order to evaluate performance')
    parser.add_argument('--config_filename', type=str, default=default_config_filename, help='configuration file to use')
    parser.add_argument('--before', action='store_true', help='Whether to show the object before the action')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = YamlConfig(args.config_filename)

    if config['debug']:
        random.seed(SEED)
        np.random.seed(SEED)

    env = GraspingEnv(config, config['vis'])
    test_multi_push(env)
