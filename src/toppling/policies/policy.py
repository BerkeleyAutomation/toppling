import numpy as np
from copy import deepcopy
import logging
import sys

from trimesh import sample 
from dexnet.envs import MultiEnvPolicy, DexNetGreedyGraspingPolicy
from autolab_core import YamlConfig
from toppling.models import TopplingModel

class TopplingPolicy(MultiEnvPolicy):
    def __init__(self, grasping_policy_config_filename):
        MultiEnvPolicy.__init__(self)
        config = YamlConfig(grasping_policy_config_filename)
        database_config = config['policy']['database']
        params_config = config['policy']['params']
        self.grasping_policy = DexNetGreedyGraspingPolicy(database_config, params_config)
    
    def set_environment(self, environment):
        MultiEnvPolicy.set_environment(self, environment)
        self.grasping_policy.set_environment(environment)

    def action(self, state):
        """
        returns the push vertex and direction which maximizes the grasp quality after topping
        the object at that push vertex

        Parameters
        ----------
        state : :obj:`ObjectState`
        """
        com = state.T_obj_world.translation
        mesh = deepcopy(state.mesh).apply_transform(state.T_obj_world.matrix)
        mesh.fix_normals()
        vertices, face_ind = sample.sample_surface_even(mesh, 1000)
        normals = mesh.face_normals[face_ind]
        push_directions = -deepcopy(normals)
        
        toppling_model = TopplingModel(state)
        probabilities = toppling_model.predict(vertices, normals, push_directions, use_sensitivity=True)
        final_poses = toppling_model.map_edge_to_pose()
        toppling_model.com_projected_on_edges.append(toppling_model.com)
        return {
            'vertices': vertices, 
            'probabilities': probabilities, 
            'final_poses': final_poses,
            'bottom_points': toppling_model.bottom_points,
            'com': toppling_model.com
        }
        
