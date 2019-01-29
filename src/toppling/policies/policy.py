import numpy as np
from copy import deepcopy
import logging
import sys
from time import time

from trimesh import sample 
from dexnet.envs import MultiEnvPolicy, DexNetGreedyGraspingPolicy, LinearPushAction
from autolab_core import YamlConfig, RigidTransform
from toppling.models import TopplingModel
from toppling import normalize, up

class TopplingPolicy(MultiEnvPolicy):
    def __init__(self, grasping_policy_config_filename, use_sensitivity=True, num_samples=1000):
        MultiEnvPolicy.__init__(self)
        grasping_config = YamlConfig(grasping_policy_config_filename)
        database_config = grasping_config['policy']['database']
        params_config = grasping_config['policy']['params']
        self.grasping_policy = DexNetGreedyGraspingPolicy(database_config, params_config)
        self.use_sensitivity = use_sensitivity
        self.num_samples = num_samples
    
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

    # def get_best_topple_vertex(self, quality_increases):
    #     # vertex probs for final poses with best grasp quality
        

    def action(self, state):
        """
        returns the push vertex and direction which maximizes the grasp quality after topping
        the object at that push vertex

        Parameters
        ----------
        state : :obj:`ObjectState`
        """
        policy_start = time()
        # centering around world frame origin
        state.T_obj_world.translation[:2] = np.array([0,0])
        mesh = deepcopy(state.mesh).apply_transform(state.T_obj_world.matrix)

        mesh.fix_normals()
        vertices, face_ind = sample.sample_surface_even(mesh, self.num_samples)
        # Cut out vertices that are too close to the ground
        z_comp = vertices[:,2]
        thresh = .15
        valid_vertex_ind = z_comp > (1-thresh)*np.min(z_comp) + thresh*np.max(z_comp)
        vertices, face_ind = vertices[valid_vertex_ind], face_ind[valid_vertex_ind]

        normals = mesh.face_normals[face_ind]
        push_directions = -deepcopy(normals)
        
        toppling_model = TopplingModel(state)
        poses, vertex_probs = toppling_model.predict(
            vertices, 
            normals, 
            push_directions, 
            use_sensitivity=self.use_sensitivity
        )

        T_old = deepcopy(state.obj.T_obj_world)
        grasp_start = time()
        qualities = np.array([self.quality(state, pose.T_obj_table) for pose in poses])
        print 'grasp quality time:', time() - grasp_start
        # quality_increases = (quality_increases + 1 - quality_increases[0]) / 2.0
        # quality_increases = np.maximum(quality_increases - quality_increases[0], 0)

        #quality_increases = qualities - np.amin(qualities)
        #quality_increases = quality_increases / (np.amax(quality_increases) + 1e-5)
        quality_increases = (qualities - qualities[0]) / 2 + .5
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
        print 'Total Policy Time:', time() - policy_start
        
        return LinearPushAction(
            start_pose,
            end_pose,
            metadata={
                'vertices': vertices, 
                'vertex_probs': vertex_probs,
                'topple_probs': topple_probs,
                'quality_increases': quality_increases,
                'qualities': qualities[1:],
                'current_pose': state.T_obj_world,
                'current_quality': qualities[0],
                'final_poses': [stable_pose.T_obj_table for stable_pose in toppling_model.final_poses[1:]], # remove the first pose which corresponds to "no topple"
                'bottom_points': toppling_model.bottom_points,
                'com': toppling_model.com,
                'final_pose_ind': final_pose_ind
            }
        )
