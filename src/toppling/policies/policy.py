import numpy as np
from copy import deepcopy
import logging
import sys
from time import time
from abc import ABCMeta, abstractmethod
import networkx as nx

from trimesh import sample 
from dexnet.envs import MultiEnvPolicy, DexNetGreedyGraspingPolicy, LinearPushAction, NoActionFoundException
from autolab_core import YamlConfig, RigidTransform
from toppling.models import TopplingModel
from toppling import normalize, up, is_equivalent_pose, camera_pose

class TopplePolicy(MultiEnvPolicy):
    __metaclass__ = ABCMeta

    def __init__(self, config, use_sensitivity=True, num_samples=1000):
        """
        config : :obj:`autolab_core.YamlConfig`
            configuration with toppling parameters
        use_sensitivity : bool
            Whether to run multiple trials per vertex to get a probability estimate
        num_samples : int
            how many vertices to sample on the object surface
        """
        MultiEnvPolicy.__init__(self)
        grasping_config = YamlConfig(config['grasping_policy_config_filename'])
        self.grasping_policy = DexNetGreedyGraspingPolicy(
            grasping_config['policy']['database'], 
            grasping_config['policy']['params']
        )

        policy_params = config['policy_params']
        self.use_sensitivity = use_sensitivity
        self.num_samples = num_samples
        self.thresh = policy_params['thresh']
        self.log = policy_params['log']
        
        self.toppling_model = TopplingModel(config['model_params'])
    
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
        except NoActionFoundException as e:
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

    def topple_action(self, state):
        mesh = state.mesh.copy().apply_transform(state.T_obj_world.matrix)

        mesh.fix_normals()
        vertices, face_ind = sample.sample_surface_even(mesh, self.num_samples)
        # Cut out vertices that are too close to the ground
        z_comp = vertices[:,2]
        valid_vertex_ind = z_comp > (1-self.thresh)*np.min(z_comp) + self.thresh*np.max(z_comp)
        vertices, face_ind = vertices[valid_vertex_ind], face_ind[valid_vertex_ind]

        normals = mesh.face_normals[face_ind]
        push_directions = -deepcopy(normals)

        self.toppling_model.load_object(state)
        poses, vertex_probs, min_required_forces = self.toppling_model.predict(
            vertices, 
            normals, 
            push_directions, 
            use_sensitivity=self.use_sensitivity
        )
        from ambicore.visualization import Visualizer3D as vis3d
        j = 214
        a = j*self.toppling_model.n_trials
        b = (j+1)*self.toppling_model.n_trials
        # for point in self.toppling_model.vertices[a:b]:
        #    vis3d.points(Point(point), scale=.001)
        for i in range(a, b):
            start = self.toppling_model.vertices[i]
            end = start - .01*self.toppling_model.push_directions[i]
            vis3d.plot([start, end], color=[0,1,0], radius=.0002)
        #vis3d.points(Point(vertices[j]), scale=.001)
        #self.env.render_3d_scene()
        vis3d.mesh(self.env.state.mesh, self.env.state.T_obj_world.matrix, color=self.env.state.color)
        vis3d.show(starting_camera_pose=camera_pose())

        grasp_start = time()
        T_old = deepcopy(state.obj.T_obj_world)
        qualities = np.array([self.quality(state, pose) for pose in poses])
        qualities = np.random.random(len(poses))
        state.obj.T_obj_world = T_old
        if self.log:
            print 'grasp quality time:', time() - grasp_start
            print 'qualities', qualities

        quality_increases = qualities - np.amin(qualities)
        quality_increases = quality_increases / (np.amax(quality_increases) + 1e-5)

        topple_probs = np.sum(vertex_probs[:,1:], axis=1)
        quality_increases = vertex_probs.dot(quality_increases)
        final_pose_ind = np.argmax(vertex_probs, axis=1)
        return {
            # Per vertex quantities
            'vertices': vertices, 
            'normals': normals,
            'vertex_probs': vertex_probs,
            'topple_probs': topple_probs,
            'final_pose_ind': final_pose_ind,
            'quality_increases': quality_increases,
            'min_required_forces': min_required_forces,

            # Per edge quantities
            'qualities': qualities[1:],
            'current_quality': qualities[0],
            'final_poses': poses[1:], # remove the first pose which corresponds to "no topple"
            'bottom_points': self.toppling_model.bottom_points,
        }

    @abstractmethod
    def action(self, state):
        """
        returns the push vertex and direction which maximizes the grasp quality after topping
        the object at that push vertex

        Parameters
        ----------
        state : :obj:`ObjectState`
        """
        pass

class SingleTopplePolicy(TopplePolicy):
    def action(self, state, env):
        """
        returns the push vertex and direction which maximizes the grasp quality after topping
        the object at that push vertex

        Parameters
        ----------
        state : :obj:`ObjectState`
        """
        self.env = env
        policy_start = time()
        orig_pose = deepcopy(state.T_obj_world)

        topple_action_metadata = self.topple_action(state)
        quality_increases = topple_action_metadata['quality_increases']
        min_required_forces = topple_action_metadata['min_required_forces']
        vertices = topple_action_metadata['vertices']
        normals = topple_action_metadata['normals']
        
        best_topple_vertices = np.arange(len(quality_increases))[quality_increases == np.amax(quality_increases)]
        least_force = np.argmin(min_required_forces[best_topple_vertices])
        best_ind = best_topple_vertices[least_force]
        start_position = vertices[best_ind] + normals[best_ind] * .015
        end_position = vertices[best_ind] - normals[best_ind] * .04
        R_push = self.get_hand_pose(start_position, end_position)
        
        start_pose = RigidTransform(
            rotation=R_push,
            translation=start_position,
            from_frame='grasp',
            to_frame='world'
        )
        end_pose = RigidTransform(
            rotation=R_push,
            translation=end_position,
            from_frame='grasp',
            to_frame='world'
        )
        print 'Total Policy Time:', time() - policy_start
        state.obj.T_obj_world = orig_pose
        
        topple_action_metadata['best_ind'] = best_ind
        return LinearPushAction(
            start_pose,
            end_pose,
            metadata=topple_action_metadata
        )

class MultiTopplePolicy(TopplePolicy):
    def __init__(self, config, use_sensitivity=True, num_samples=1000):
        """
        config : :obj:`autolab_core.YamlConfig`
            configuration with toppling parameters
        use_sensitivity : bool
            Whether to run multiple trials per vertex to get a probability estimate
        num_samples : int
            how many vertices to sample on the object surface
        """
        self.gamma = .95
        TopplePolicy.__init__(self, config, use_sensitivity, num_samples)

    def add_all_nodes_old(self, node_id, metadata, check_duplicates=True):
        """
        """
        if self.log:
            print self.node_idx
        edge_alphas = []
        for pose_ind, (pose, quality) in enumerate(zip(metadata['final_poses'], metadata['qualities'])):
            # Check if this pose exists in the graph already
            already_exists = False
            if check_duplicates:
                num_existing_nodes = len(self.G.nodes())
                for j, node in self.G.nodes(data=True):
                    if is_equivalent_pose(pose, node['pose']):
                        already_exists = True
                        break
            if not already_exists:
                self.G.add_node(self.node_idx, pose=pose, gq=quality, node_type='state')
                to_node_idx = self.node_idx
                #self.G.add_edge(node_id, self.node_idx)
                if self.log:
                    print 'edge from {} to {}={}'.format(node_id, self.node_idx, np.clip(np.max(metadata['vertex_probs'][:,pose_ind]), 0, 1))
                self.node_idx += 1
            else:
                to_node_idx = j
                if self.log:
                    print 'edge from {} to {}={}'.format(node_id, j, np.clip(np.max(metadata['vertex_probs'][:,pose_ind]), 0, 1))
                #self.G.add_edge(node_id, j)
            self.G.add_edge(node_id, to_node_idx)
            edge_alphas.append(np.clip(np.max(metadata['vertex_probs'][:,pose_ind]), 0, 1))
        return edge_alphas

    def add_all_nodes(self, node_id, metadata, check_duplicates=True):
        """
        """
        edge_alphas = []
        vertex_probs = metadata['vertex_probs']
        metadata_to_graph_mapping = []
        for pose, quality in zip(metadata['final_poses'], metadata['qualities']):
            # Check if this pose exists in the graph already
            already_exists = False
            if check_duplicates:
                num_existing_nodes = len(self.G.nodes())
                for i, node in self.G.nodes(data=True):
                    if node['node_type'] == 'state' and is_equivalent_pose(pose, node['pose']):
                        already_exists = True
                        break
            if not already_exists:
                self.G.add_node(self.node_idx, pose=pose, gq=quality, value=quality, node_type='state')
                metadata_to_graph_mapping.append(self.node_idx)
                self.node_idx += 1
            else:
                metadata_to_graph_mapping.append(i)
        
        
        for pose_ind in range(len(metadata['final_poses'])):
            best_action = vertex_probs[np.argmax(vertex_probs[:,pose_ind])]
            #self.G.add_node(self.action_node_idx, best_action=best_action, value=0, node_type='action')
            action_node_idx = str(node_id)+str(metadata_to_graph_mapping[pose_ind])
            self.G.add_node(action_node_idx, best_action=best_action, value=0, node_type='action')
            self.G.add_edge(node_id, action_node_idx)
            edge_alphas.append(1)
            for prob, next_node_id, in zip(best_action, metadata_to_graph_mapping):
                if prob != 0.0:
                    self.G.add_edge(action_node_idx, next_node_id, prob=prob)
                    edge_alphas.append(np.clip(prob, 0, 1))
            #self.action_node_idx += 1
            
        return edge_alphas

    def value_iteration(self):
        if self.log:
            print 'values', self.G.nodes('value'), '\n'
        while True:
            unchanged = True
            for node_id, node in self.G.nodes(data=True):
                if node['node_type'] == 'state':
                    values = [node['value']]
                    for action in self.G.neighbors(node_id):
                        action = self.G.nodes[action]
                        values.append(action['value'])
                    node['value'] = np.max(values)
                else: # node is action node
                    #print self.G.edges(node_id) # [(1000,1)
                    q_value = 0
                    for next_state in self.G.neighbors(node_id):
                        transition_prob = self.G.edges[node_id, next_state]['prob']
                        next_state = self.G.nodes[next_state]
                        q_value += self.gamma * transition_prob * next_state['value']
                    if q_value > node['value'] * 1.05:
                        unchanged = False
                    node['value'] = q_value
                    #sys.exit()
            if self.log:
                print 'values', self.G.nodes('value'), '\n'
            if unchanged:
                break
        if self.log:
            print 'gq', self.G.nodes('gq'), '\n\n'

    def action(self, state):
        """
        returns the push vertex and direction which maximizes the grasp quality after topping
        the object at that push vertex

        Parameters
        ----------
        state : :obj:`ObjectState`
        """
        add_nodes = self.add_all_nodes
        policy_start = time()
        orig_pose = deepcopy(state.T_obj_world)

        original_action_metadata = self.topple_action(state)

        self.G = nx.DiGraph()
        current_quality = original_action_metadata['current_quality']
        self.G.add_node(0, pose=state.T_obj_world, gq=current_quality, value=current_quality, node_type='state')
        self.node_idx = 1
        self.action_node_idx = 1000
        self.edge_alphas = add_nodes(0, original_action_metadata, check_duplicates=False)

        nodes = iter(self.G.nodes(data=True))
        already_visited = [0]
        while True:
            node_id, node = next(nodes, (None, None))
            if node_id is None:
                break
            if node['node_type'] == 'action':
                already_visited.append(node_id)
                continue
            if node_id in already_visited:
                continue
            if self.log:
                print '\nPose Ind: {}'.format(node_id)
            state.obj.T_obj_world = node['pose']
            metadata = self.topple_action(state)
            new_edge_alphas = add_nodes(node_id, metadata)
            self.edge_alphas.extend(new_edge_alphas)
            nodes = iter(self.G.nodes(data=True))
            already_visited.append(node_id)
            # break

        self.value_iteration()
        #print self.edge_alphas
        state.obj.T_obj_world = orig_pose
        print 'Total Policy Time:', time() - policy_start
        return None
