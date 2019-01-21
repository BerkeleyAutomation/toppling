import numpy as np
from copy import deepcopy
import logging
import sys

from trimesh import sample 
from dexnet.envs import MultiEnvPolicy, DexNetGreedyGraspingPolicy, LinearPushAction
from autolab_core import YamlConfig, RigidTransform
from toppling.models import TopplingModel
from toppling import normalize, up

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

    def get_hand_pose(self, start, end):
        """
        Computes the pose of the hand to be perpendicular to the direction of the push

        Parameters
        ----------
        start : 3x' :obj:`numpy.ndarray`
        end : 3x' :obj:`numpy.ndarray`

        Returns
        -------
        3x3 :obj:`numpy.ndarray`
            3D Rotation Matrix
        """
        z = normalize(end - start)
        y = normalize(np.cross(z, -up))
        x = normalize(np.cross(z, -y))
        return np.hstack((x.reshape((-1,1)), y.reshape((-1,1)), z.reshape((-1,1))))

    def action(self, state):
        """
        returns the push vertex and direction which maximizes the grasp quality after topping
        the object at that push vertex

        Parameters
        ----------
        state : :obj:`ObjectState`
        """
        # centering around world frame origin
        state.T_obj_world.translation[:2] = np.array([0,0])
        mesh = deepcopy(state.mesh).apply_transform(state.T_obj_world.matrix)

        mesh.fix_normals()
        vertices, face_ind = sample.sample_surface_even(mesh, 1000)
        normals = mesh.face_normals[face_ind]
        push_directions = -deepcopy(normals)
        
        toppling_model = TopplingModel(state)
        poses, vertex_probs = toppling_model.predict(vertices, normals, push_directions, use_sensitivity=False)

        T_old = deepcopy(state.obj.T_obj_world)
        quality_increases = np.array([self.quality(state, pose.T_obj_table) for pose in poses])
        # quality_increases = (quality_increases + 1 - quality_increases[0]) / 2.0
        # quality_increases = np.maximum(quality_increases - quality_increases[0], 0)
        quality_increases = quality_increases - np.amin(quality_increases)
        quality_increases = quality_increases / (np.amax(quality_increases) + 1e-5)
        state.obj.T_obj_world = T_old

        topple_probs = np.sum(vertex_probs[:,1:], axis=1)
        quality_increases = vertex_probs.dot(quality_increases)
        final_pose_ind = np.argmax(vertex_probs, axis=1)

        best_topple_vertices = np.arange(len(quality_increases))[quality_increases == np.amax(quality_increases)]
        best_ind = best_topple_vertices[0]
        start_position = vertices[best_ind] + normals[best_ind] * .01
        end_position = vertices[best_ind] - normals[best_ind] * .01
        R_push = self.get_hand_pose(start_position, end_position)
        
        start_pose = RigidTransform(
            rotation=R_push,
            translation=start_position,
            from_frame='grasp',
            to_frame='world'
        )
        end_pose = RigidTransform(
            rotation=R_push,
            translation=start_position,
            from_frame='grasp',
            to_frame='world'
        )
        
        return LinearPushAction(
            start_pose,
            end_pose,
            metadata={
                'vertices': vertices, 
                'topple_probs': topple_probs,
                'quality_increases': quality_increases,
                'final_poses': toppling_model.final_poses[1:], # remove the first pose which corresponds to "no topple"
                'bottom_points': toppling_model.bottom_points,
                'com': toppling_model.com,
                'final_pose_ind': final_pose_ind
            }
        )
