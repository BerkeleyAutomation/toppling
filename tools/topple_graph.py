import argparse
import numpy as np
import os
import matplotlib as mpl
from copy import deepcopy
# mpl.use('GTK')
import matplotlib.pyplot as plt
import networkx as nx
import random

from autolab_core import Point, RigidTransform, YamlConfig, TensorDataset
from dexnet.constants import *
from dexnet.envs import GraspingEnv, LinearPushAction
from dexnet.visualization import DexNetVisualizer3D as vis3d
from toppling.policies import TopplingPolicy
from toppling import stable_pose

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

def add_all_nodes(G, node_id, metadata, check_duplicates=True):
    """
    """
    i = len(G.nodes())
    labels, edge_alphas = {}, []
    for pose_ind, (pose, quality) in enumerate(zip(metadata['final_poses'], metadata['qualities'])):
        # Check if this pose exists in the graph already
        already_exists = False
        if check_duplicates:
            curr_stable_pose = stable_pose(pose)
            num_existing_nodes = len(G.nodes())
            for j, node in G.nodes(data=True):
                if curr_stable_pose == stable_pose(node['pose']):
                    already_exists = True
                    break
        if not already_exists:
            G.add_node(i, pose=pose)
            labels[i] = 'Pose: {}\nGQ: {}'.format(i, quality)
            G.add_edge(node_id, i)
            i += 1
        else:
            G.add_edge(node_id, j)
        edge_alphas.append(np.clip(np.max(metadata['vertex_probs'][:,pose_ind]), 0, 1))
    return labels, edge_alphas

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
    policy = TopplingPolicy(config['policy']['grasping_policy_config_filename'], use_sensitivity=True, num_samples=800)
    
    if config['debug']:
        random.seed(SEED)
        np.random.seed(SEED)

    env = GraspingEnv(config, config['vis'])
    env.reset()
    policy.set_environment(env.environment)
    action = policy.action(env.state)
    print 'qualities', action.metadata['qualities']

    # Visualizing Topple Graph
    G = nx.DiGraph()
    G.add_node(0, pose=action.metadata['current_pose'])
    labels = {0:'Pose: 0\nGQ: {}'.format(action.metadata['current_quality'])}
    new_labels, edge_alphas = add_all_nodes(G, 0, action.metadata, check_duplicates=False)
    print edge_alphas
    labels.update(new_labels)

    nodes = iter(G.nodes(data=True))
    already_visited = [0]
    while True:
        node_id, node = next(nodes, (None, None))
        if node_id is None:
            break
        if node_id in already_visited:
            node_id += 1
            continue
        print '\nPose Ind: {}'.format(node_id)
        env.state.obj.T_obj_world = node['pose']
        action = policy.action(env.state)
        new_labels, new_edge_alphas = add_all_nodes(G, node_id, action.metadata)
        labels.update(new_labels)
        edge_alphas.extend(new_edge_alphas)
        nodes = iter(G.nodes(data=True))
        already_visited.append(node_id)

    pos = nx.layout.spring_layout(G)
    node_sizes = [3 + 10 * i for i in range(len(G))]
    M = G.number_of_edges()
    edge_colors = [M/2.0] * M

    nodes = nx.draw_networkx_nodes(G, pos, node_color='blue')
    edges = nx.draw_networkx_edges(G, pos, arrowstyle='->',
                                   arrowsize=10,
                                   edge_cmap=plt.cm.Blues, width=2
    )
    nx.draw_networkx_labels(G, pos, labels, font_color='red')

    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])

    ax = plt.gca()
    ax.set_axis_off()
    plt.show()
    

