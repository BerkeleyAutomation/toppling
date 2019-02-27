import argparse
import numpy as np
import os
import matplotlib as mpl
from copy import deepcopy
import matplotlib.pyplot as plt
import networkx as nx
import random

from autolab_core import Point, RigidTransform, YamlConfig, TensorDataset
from dexnet.constants import *
from dexnet.envs import GraspingEnv, LinearPushAction
from dexnet.visualization import DexNetVisualizer3D as vis3d
from toppling.policies import SingleTopplePolicy, MultiTopplePolicy
from toppling import camera_pose

SEED = 107

DARK_BLUE = np.array([0,.298,.427])
LIGHT_BLUE = np.array([.615,.776,.882])
CAMERA_POSE = camera_pose()

def show_graph(G, edge_alphas):
    #pos = nx.layout.spring_layout(G)
    pos = nx.layout.circular_layout(G)

    for node_id, node in G.nodes(data=True):
        if node['node_type'] == 'state':
            node_plt = nx.draw_networkx_nodes(
                G, 
                pos, 
                nodelist=[node_id], 
                node_shape='s', 
                node_size=100, # 500
                node_color=DARK_BLUE, 
                linewidth=10, 
                label='Pose: {}\nGQ: {}\n'.format(node_id, node['gq'])
            )
            node_plt.set_edgecolor('k')
        else:
            node_plt = nx.draw_networkx_nodes(
                G, 
                pos, 
                nodelist=[node_id], 
                node_shape='s', 
                node_size=100, # 500
                node_color=LIGHT_BLUE, 
                linewidth=10,
            )
            node_plt.set_edgecolor('k')

    edges = nx.draw_networkx_edges(G, pos, arrowstyle='->',
                                   arrowsize=10, width=5 # 40
    )
    # edge_labels=dict([((u,v,),d['weight'])
    #          for u,v,d in G.edges(data=True)])
    nx.draw_networkx_labels(G, pos, font_size=20, font_color=LIGHT_BLUE)
    # nx.draw_networkx_edge_labels(G, pos, labels=edge_labels)

    for i in range(G.number_of_edges()):
        edges[i].set_alpha(edge_alphas[i])

    ax = plt.gca()
    ax.set_axis_off()
    plt.legend()
    plt.show()

def new_graph(G):
    for node_id, node in G.nodes(data=True):
        if node['node_type'] == 'action':
            continue
        print 'Pose {}, GQ: {}'.format(node_id, node['gq'])
        env.state.obj.T_obj_world = node['pose']
        env.render_3d_scene()
        vis3d.show(starting_camera_pose=CAMERA_POSE)

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
    policy = MultiTopplePolicy(config, use_sensitivity=True, num_samples=800)
    
    if config['debug']:
        random.seed(SEED)
        np.random.seed(SEED)

    env = GraspingEnv(config, config['vis'])
    env.reset()
    #env.state.material_props._color = np.array([0.86274509803] * 3)
    env.state.material_props._color = np.array([0.5] * 3)

    if args.before:
        env.render_3d_scene()
        color = np.array([0,0,0])
        vert = np.array([-0.01091172,  0.02806294, 0.06962403])
        normal = vert + .01*np.array([-0.84288757, -0.3828792,  0.37807943])
        vis3d.points(Point(vert), scale=.001, color=color)
        vis3d.points(Point(normal), scale=.001, color=np.array([1,0,0]))
        vis3d.show(starting_camera_pose=CAMERA_POSE)


    policy.set_environment(env.environment)
    policy.action(env.state)
    #show_graph(policy.G, policy.edge_alphas)
    new_graph(policy.G)
