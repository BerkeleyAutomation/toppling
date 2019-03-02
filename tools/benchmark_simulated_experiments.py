import argparse
import numpy as np
import os
import colorsys
import random
import logging
from time import time
from copy import deepcopy

from autolab_core import Point, RigidTransform, YamlConfig, TensorDataset
from dexnet.constants import *
from dexnet.envs import GraspingEnv
from dexnet.visualization import DexNetVisualizer3D as vis3d
from toppling.policies import SingleTopplePolicy, MultiTopplePolicy, RandomTopplePolicy
from toppling import is_equivalent_pose, camera_pose

SEED = 107
CAMERA_POSE = camera_pose()

def to_rigid(mat):
    rot, trans = RigidTransform.rotation_and_translation_from_matrix(mat)
    return RigidTransform(rot, trans, 'obj', 'world')

def render_3d_scene(env):
    vis3d.mesh(env.state.mesh,
            env.state.T_obj_world.matrix,
            color=env.state.color)

def forward_sim(G, policy_name, value):
    curr_node_id = 0
    curr_node = G.nodes[curr_node_id]
    logger.info(policy_name + ' Original Quality: '+str(curr_node['gq']))
    path = [0]
    while True:
        best_action, best_q = None, 0
        for action_id in G.neighbors(curr_node_id):
            action = G.nodes[action_id]
            if action[value] > best_q:
                best_q = action[value]
                best_action = action_id
        if best_q <= curr_node['gq']:
            break
        # execute best action
        next_state_edges = filter(lambda (u,a,b): u == best_action, G.edges.data())
        _, next_state_ids, probs = zip(*next_state_edges)
        probs = map(lambda prob: prob['prob'], probs)
        curr_node_id = np.random.choice(next_state_ids, p=probs/np.sum(probs))
        curr_node = G.nodes[curr_node_id]
        path.append(curr_node_id)
    logger.info(policy_name + ' Final Quality: '+str(curr_node['gq']))
    return path

def forward_sim_random(env, policy):
    curr_q = policy.quality(env.state)
    logger.info('Random Original Quality: '+str(curr_q))

    obj_config = config['state_space']['object']
    all_poses, _ = env.state.obj.mesh.compute_stable_poses(
        sigma=obj_config['stp_com_sigma'],
        n_samples=obj_config['stp_num_samples'],
        threshold=obj_config['stp_min_prob']
    )
    orig_pose = deepcopy(env.state.obj.T_obj_world)
    best_q = np.max([policy.quality(env.state, to_rigid(pose)) for pose in all_poses])   
    env.state.obj.T_obj_world = orig_pose
    
    num_failed_actions = 0
    path_length = 0
    planning_time = 0
    while num_failed_actions < 3:
        if curr_q == best_q or path_length == 10:
            break

        planning_time -= time()
        action = policy.action(env.state)
        planning_time += time()

        policy.toppling_model.load_object(env.state)
        poses, vertex_probs, _ = policy.toppling_model.predict(
            [action.metadata['vertex']], 
            [action.metadata['normal']], 
            [-action.metadata['normal']], 
            use_sensitivity=policy.use_sensitivity
        )
        vertex_probs = vertex_probs[0]

        pose_ind = np.random.choice(np.arange(len(poses)), p=vertex_probs/np.sum(vertex_probs))
        if pose_ind == 0:
            num_failed_actions += 1
        else:
            num_failed_actions = 0
            env.state.T_obj_world = poses[pose_ind]
            curr_q = policy.quality(env.state)
        path_length += 1
    logger.info('Random Final Quality: '+str(curr_q))
    return path_length, planning_time

def test_multi_push(env):
    env = GraspingEnv(config, config['vis'])
    policy = MultiTopplePolicy(config, use_sensitivity=True)
    config['model']['load'] = 0
    rand_policy = RandomTopplePolicy(config, use_sensitivity=True)
    while True:
        env.reset()
        env.state.material_props._color = np.array([0.5] * 3)
        policy.set_environment(env.environment)
        rand_policy.set_environment(env.environment)
        
        if args.before:
            vis3d.figure()
            env.render_3d_scene()
            vis3d.show(starting_camera_pose=CAMERA_POSE)
        planning_time = policy.action(env.state)
        logger.info(env.state.obj.key)
        logger.info('Value Iteration')
        path = forward_sim(policy.G, 'Value Iteration', 'value')
        logger.info('Value Iteration Path: '+str(path))
        logger.info('Value Iteration Path Length: '+str(len(path) - 1))
        logger.info('Value Iteration Planning Time: '+str(planning_time))
        logger.info('Value Iteration No Actions: '+str(len(policy.G.nodes()) == 1))
        logger.info('Value Iteration Already Best: '+str(len(policy.G.nodes()) > 1 and len(path) == 1))
        logger.info('Greedy')
        path = forward_sim(policy.G, 'Greedy', 'single_push_q')
        logger.info('Greedy Path: '+str(path))
        logger.info('Greedy Path Length: '+str(len(path) - 1))
        logger.info('Greedy Planning Time: '+str(np.sum([policy.G.nodes[node_id]['planning_time'] for node_id in path])))
        logger.info('Greedy No Actions: '+str(len(policy.G.nodes()) == 1))
        logger.info('Greedy Already Best: '+str((len(policy.G.nodes()) > 1 and len(path) == 1)))
        a = time()
        path_length, planning_time = forward_sim_random(env, rand_policy)
        logger.info('Random Path Length: '+str(path_length))
        logger.info('Random Planning Time: '+str(planning_time))
        logger.info('\n')

def parse_args():
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/benchmark_topple_policy_graspingenv.yaml'
    )
    parser = argparse.ArgumentParser(description='Rollout a policy for bin picking in order to evaluate performance')
    parser.add_argument('--config_filename', type=str, default=default_config_filename, help='configuration file to use')
    parser.add_argument('--before', action='store_true', help='Whether to show the object before the action')
    parser.add_argument('-logfile', type=str, default='/home/chriscorrea14/toppling_simulated'+str(int(time()))+'.log')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = YamlConfig(args.config_filename)

    if config['debug']:
        random.seed(SEED)
        np.random.seed(SEED)
    logger = logging.getLogger('toppling')
    hdlr = logging.FileHandler(args.logfile)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    env = GraspingEnv(config, config['vis'])
    test_multi_push(env)
