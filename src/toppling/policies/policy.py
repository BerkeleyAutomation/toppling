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
        grasping_config = YamlConfig(grasping_policy_config_filename)
        database_config = grasping_config['policy']['database']
        params_config = grasping_config['policy']['params']
        self.grasping_policy = DexNetGreedyGraspingPolicy(database_config, params_config)
    
    def set_environment(self, environment):
        MultiEnvPolicy.set_environment(self, environment)
        self.grasping_policy.set_environment(environment)

    def quality(self, state, T_obj_world):
        """
        Computes the grasp quality of the object in the passed in state

        Parameters
        ----------
        state : :obj:`ObjectState`
        T_obj_world : :obj:`RigidTransform`

        Returns
        -------
        float
            highest grasp quality of the object, or 0 if no grasps are found
        """
        state.obj.T_obj_world = T_obj_world
        try:
            return self.grasping_policy.action(state).q_value
        except:
            return 0

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
        poses, vertex_probs = toppling_model.predict(vertices, normals, push_directions, use_sensitivity=False)

        T_old = deepcopy(state.obj.T_obj_world)
        quality_increases = np.array([self.quality(state, pose.T_obj_table) for pose in poses])
        print 'Current Quality:', quality_increases
        quality_increases = (quality_increases + 1 - quality_increases[0]) / 2.0
        state.obj.T_obj_world = T_old

        #probabilities = np.sum(vertex_probs, axis=1)
        probabilities = vertex_probs.dot(quality_increases)
        return {
            'vertices': vertices, 
            'probabilities': probabilities, 
            'final_poses': toppling_model.final_poses,
            'bottom_points': toppling_model.bottom_points,
            'com': toppling_model.com
        }
        
