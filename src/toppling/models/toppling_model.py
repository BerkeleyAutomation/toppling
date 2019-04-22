import numpy as np
from scipy.spatial import ConvexHull
import math
import json
from copy import deepcopy
from time import time
from itertools import groupby

from autolab_core import RigidTransform, PointCloud, Point, TensorDataset
from toppling import normalize, stable_pose, pose_diff, is_equivalent_pose, up

class TopplingModel():
    def __init__(self, config, obj=None):
        """
        Parameters
        ----------
        config : :obj:`autolab_core.YamlConfig`
            configuration with toppling parameters
        obj : :obj:`GraspableObject3D`
            object to load
        """
        self.fraction_before_short_circuit = config['fraction_before_short_circuit']
        self.num_approx = config['num_approx']
        
        self.ground_friction_coeff = config['ground_friction_coeff']
        self.ground_friction_sigma = config['ground_friction_sigma']
        self.finger_friction_coeff = config['finger_friction_coeff']
        self.finger_friction_sigma = config['finger_friction_sigma']
        
        self.finger_sigma = config['finger_sigma'] # noise for finger position
        self.push_direction_sigma = config['push_direction_sigma'] # noise for push direction
        self.obj_rot_sigma = config['obj_rot_sigma']
        self.n_trials = config['n_trials'] # if you choose to use sensitivity
        self.max_force = config['max_force']
        self.log = config['log']
        self.baseline = config['baseline']
        if obj != None:
            self.load_object(obj)

    def readable_str(self):
        return 'ground friction~N({:.4f}, {}), finger friction~N({:.4f}, {}), c_f~N(c_f, {:.7f}I), f_f~N(f_f, {:.4f}I), R_o~N(I, R_{:.4f})'.format(
           self.ground_friction_coeff, self.ground_friction_sigma,
           self.finger_friction_coeff, self.finger_friction_sigma,
           self.finger_sigma, self.push_direction_sigma,
           self.obj_rot_sigma
        )

    def __str__(self):
        #return 'Toppling Model, ground_friction_coeff: {}, ground_friction_sigma: {}, finger_friction_coeff: {}, finger_friction_sigma: {}'.format(
        #    self.ground_friction_coeff, self.ground_friction_sigma,
        #    self.finger_friction_coeff, self.finger_friction_sigma
        #)
        return ', '.join([
            '$\mu_T\sim\mathcal{{N}}({:.4f}, {})'.format(self.ground_friction_coeff, self.ground_friction_sigma),
            '\mu_f\sim\mathcal{{N}}({:.4f}, {})'.format(self.finger_friction_coeff, self.finger_friction_sigma),
            '\mathbf{{c}}_f\sim\mathcal{{N}}(\mathbf{{c}}_f, {:.2e}\cdot I)'.format(self.finger_sigma),
            '\mathbf{{f}}_f\sim\mathcal{{N}}(\mathbf{{f}}_f, {:.4}\cdot I)$'.format(self.push_direction_sigma)
        ])
            

    def load_object(self, obj):
        """
        Does a lot of the preprocessing, like calculating the bottom points, 
        and max_moment / max_tangential_forces for each edge.  This way you can
        call predict multiple times and not redo too much work

        Parameters
        ----------
        obj : :obj:`GraspableObject3D`
            object to load
        """
        self.obj = obj
        self.mesh = obj.mesh.copy().apply_transform(obj.T_obj_world.matrix)
        self.mesh.fix_normals()
        self.com = self.mesh.center_mass
        # self.com = obj.T_obj_world.translation
        # self.com = np.mean(self.mesh.vertices, axis=0)
        self.mass = 1
       
        # Finding toppling edge
        z_components = np.around(self.mesh.vertices[:,2], 3)
        self.lowest_z = np.min(z_components)
        cutoff = .02
        bottom_points = self.mesh.vertices[z_components == self.lowest_z][:,:2]
        bottom_points = bottom_points[ConvexHull(bottom_points).vertices]
        self.bottom_points = np.append(
            bottom_points,
            #ensure all points lie on plane
            self.lowest_z * np.ones(len(bottom_points)).reshape((-1,1)), 
            axis=1
        )
        #print 'bottom', self.bottom_points
        #self.bottom_points = self.bottom_points[::5]
        #print 'bottom', self.bottom_points
        
        # Turn bottom points into pairs of adjacent bottom_points
        self.edge_points = zip(self.bottom_points, np.roll(self.bottom_points,-1,axis=0))
        edge = 1
        #self.edge_points = [self.edge_points[edge]]
        #self.bottom_points = self.bottom_points[[edge,edge+1]]

        # For each edge, calculate the maximum moment
        # and maximum tangential force it can resist
        self.max_moments, self.max_tangential_forces, self.com_projected_on_edges = [], [], []
        for edge_point1, edge_point2 in self.edge_points:
            s = normalize(edge_point2 - edge_point1)
            com_projected_on_edge = (self.com - edge_point1).dot(s)*s + edge_point1
            
            offset_dist = np.linalg.norm(edge_point1 - edge_point2)
            offsets = np.linspace(0, offset_dist, self.num_approx)
            offsets_relative_to_com = offsets - np.linalg.norm(com_projected_on_edge - edge_point1)
            # This is for a non-uniform pressure distribution
            #paths = self.mesh.section_multiplane(edge_point1, s, offsets)
            #mass_per_unit = np.array([0 if paths[index] == None else paths[index].area for index in range(self.num_approx)])

            # Using a uniform pressure distribution along the contact edge (for now)
            mass_per_unit = np.array([self.mass / float(self.num_approx)] * self.num_approx)

            # pressure distribution from mass
            pressure_dist = mass_per_unit * self.mass * 9.8 / (np.sum(mass_per_unit) + offset_dist / self.num_approx) 
            self.max_moments.append(self.ground_friction_coeff * pressure_dist.dot(np.abs(offsets_relative_to_com)))
            # self.max_moments.append(self.ground_friction_coeff * mass * 9.8 * (y0**2 + y1**2) / 2) # Why doesn't this work ?????
            self.max_tangential_forces.append(self.ground_friction_coeff * (self.mass * 9.8)) # mu F_n
            self.com_projected_on_edges.append(com_projected_on_edge)

    def required_force(self, vertex, push_direction, edge_point1, edge_point2, com_projected_on_edge):
        """
        How much to press against the queried point in order to counteract gravity

        Parameters
        ----------
        vertex : 3x1 :obj:`numpy.ndarray`
        push_direction : 3x1 :obj:`numpy.ndarray`
        edge_point1 : 3x1 :obj:`numpy.ndarray`
        edge_point2 : 3x1 :obj:`numpy.ndarray`
        com_projected_on_edge : 3x1 :obj:`numpy.ndarray`

        Returns
        -------
        float
        """
        # s = normalize(edge_point2 - edge_point1)
        # vertex_projected_on_edge = (vertex - edge_point1).dot(s)*s + edge_point1 
        # f_max = normalize(np.cross(edge_point1 - vertex, s))
        # if f_max[2] < 0: # is this right??????
        #     f_max = -f_max
        # r_f = np.linalg.norm(vertex - vertex_projected_on_edge)
        # cos_push_vertex_edge_angle = push_direction.dot(f_max)
        # g_max = normalize(-np.cross(edge_point1 - self.com, s))
        # if g_max[2] > 0: # is this right??????
        #     g_max = -g_max
        # r_g = np.linalg.norm(self.com - com_projected_on_edge)
        # cos_com_edge_angle = (-up).dot(g_max)
        # 
        # return self.mass * 9.8 * cos_com_edge_angle * r_g / (cos_push_vertex_edge_angle * r_f + 1e-5)
        # [0.01282592 0.01404448 0.1]
        s = normalize(edge_point2 - edge_point1)
        vertex_projected_on_edge = (vertex - edge_point1).dot(s)*s + edge_point1
        push_projected = push_direction.dot(s)*s
        
        p_f = push_direction - push_projected
        r_f = vertex_projected_on_edge - vertex
        tau_f = np.cross(r_f, p_f)

        r_g = com_projected_on_edge - self.com
        tau_g = np.cross(r_g, self.mass * 9.8 * -up)
        # since the two torque vectors are aligned, the ratio of tau_g / tau_f
        # should be the same regardless so we just take the the x value
        return -(tau_g / (tau_f + 1e-7))[0]

    def induced_torque(
        self,
        vertex,
        push_direction, 
        edge_point1, 
        edge_point2, 
        com_projected_on_edge, 
        required_force
    ):
        """
        how much torque around the z axis (centered around com_projected_on_edge) 
        does the pushing action exert on the object

        Parameters
        ----------
        vertex : 3x1 :obj:`numpy.ndarray`
        push_direction : 3x1 :obj:`numpy.ndarray`
        edge_point1 : 3x1 :obj:`numpy.ndarray`
        edge_point2 : 3x1 :obj:`numpy.ndarray`
        com_projected_on_edge : 3x1 :obj:`numpy.ndarray`

        Returns
        -------
        float
        """
        if self.baseline:
            return 0
        r = vertex - com_projected_on_edge
        r[2] = 0
        max_z_torque_dir = normalize(np.cross(r, up))
        cos_push_vertex_z_angle = push_direction.dot(max_z_torque_dir)
        r = np.linalg.norm(r)
        return required_force * cos_push_vertex_z_angle * r
        #induced_torque = required_force * np.linalg.norm(np.cross(r, push_direction))
        # max torque increase due to the finger pressing down on the object
        

    def finger_friction_moment(self, f_z, edge_point1, edge_point2):
        """
        maximum increase in torque that can be resisted due to the finger pressing down
        on the object.  (If you press down harder, you can resist more torques, if you
        are lifting up the object, it will resist fewer torques)

        Parameters
        ----------
        f_z : float
            how much the finger would press in the downward direction
            in order to topple the object
        edge_point1 : 3x1 :obj:`numpy.ndarray`
        edge_point2 : 3x1 :obj:`numpy.ndarray`

        Returns
        -------
        float
        """
        v = self.com - edge_point1
        s = normalize(edge_point2 - edge_point1)
        com_projected_on_edge = v.dot(s) * s + edge_point1
        
        offset_dist = np.linalg.norm(edge_point1 - edge_point2)
        offsets = np.linspace(0, offset_dist, self.num_approx)
        offsets_relative_to_com = offsets - np.linalg.norm(com_projected_on_edge - edge_point1)
        downward_force_dist = np.array([f_z / float(self.num_approx)] * self.num_approx)
        return self.ground_friction_coeff * downward_force_dist.dot(np.abs(offsets_relative_to_com))

    def add_noise(self, vertices, normals, push_directions, n_trials):
        """
        adds noise to the vertex position and friction, and intersects that new position with 
        where the finger would now press against the object

        Parameters
        ----------
        vertices : nx3 :obj:`numpy.ndarray`
        normals : nx3 :obj:`numpy.ndarray`
        push_direction : nx3 :obj:`numpy.ndarray`
        n_trials : int

        Returns
        nx3 :obj`numpy.ndarray`
        nx3 :obj`numpy.ndarray`
        nx3 :obj`numpy.ndarray`
        nx1 :obj`numpy.ndarray`
        """
        vertices = np.repeat(vertices, n_trials, axis=0)
        normals = np.repeat(normals, n_trials, axis=0)
        push_directions = np.repeat(push_directions, n_trials, axis=0) \
            + np.random.normal(scale=self.push_direction_sigma, size=[len(push_directions)*n_trials, 3])
        push_directions = normalize(push_directions, axis=1)

        a = time()
        # # Add noise and find the new intersection location
        # vertices_copied = deepcopy(vertices)
        # ray_origins = vertices + .01 * normals
        # # ray_origins = vertices + np.random.normal(scale=sigma, size=vertices.shape) + .01 * normals
        # vertices, _, face_ind = mesh.ray.intersects_location(ray_origins, -normals, multiple_hits=False)
        for i in range(len(vertices)):
            # Predicted to hit ground
            if push_directions[i].dot(up) > .5:# and vertices[i][2] < .05:
                vertices[i] = np.array([0,0,0])
                normals[i] = np.array([0,0,0])
                continue
            ray_origin = vertices[i] + np.random.normal(scale=self.finger_sigma, size=3) + .001 * normals[i]
            obj_rotation = RigidTransform.rotation_from_axis_and_origin([0,0,1], self.mesh.center_mass, np.random.normal(scale=self.obj_rot_sigma)).matrix
            ray_origin = obj_rotation.dot(np.append(ray_origin, [1]))[:3]
            ray_dir = obj_rotation.dot(np.append(-normals[i], [1]))[:3]
            # ray_dir = -normals[i]

            intersect, _, face_ind = \
                self.mesh.ray.intersects_location([ray_origin], [ray_dir], multiple_hits=False)
            _, _, back_face_ind = \
                self.mesh.ray.intersects_location([ray_origin], [-ray_dir], multiple_hits=False)
            if len(face_ind) == 0 or len(back_face_ind) != 0:
                vertices[i] = np.array([0,0,0])
                normals[i] = np.array([0,0,0])
            else:
                vertices[i] = intersect[0]
                normals[i] = self.mesh.face_normals[face_ind[0]]
        ground_friction_noises = 1 + np.random.normal(scale=self.ground_friction_sigma, size=len(vertices)) / self.ground_friction_coeff
        finger_friction_noises = 1 + np.random.normal(scale=self.finger_friction_sigma, size=len(vertices)) / self.finger_friction_coeff
        if self.log:
            print 'noise time:', time() - a
        return vertices, normals, push_directions, ground_friction_noises, finger_friction_noises

    def tipping_point_rotations(self):
        tipping_point_rotations = []
        for edge, (edge_point1, edge_point2) in enumerate(self.edge_points):
            com_projected_on_edge = self.com_projected_on_edges[edge]
            s = normalize(edge_point2 - edge_point1)
            
            x = normalize(com_projected_on_edge - self.com)
            y = -normalize(np.cross(x, up))
            a = .2 if self.obj.key == "mini_dexnet~yoda" else .01
            topple_angle = math.acos(np.dot(x, -up)) + a#.01
            tipping_point_rotations.append(RigidTransform.rotation_from_axis_and_origin(y, edge_point1, topple_angle))
        return tipping_point_rotations

    def map_edge_to_pose(self, edges, tipping_point_rotations):
        """
        returns a list of poses that the object would end up in if toppled along each edge

        Returns
        -------
        :obj:`list` of :obj:`RigidTransform`
        """
        current_pose = self.obj.T_obj_world
        final_poses = []
        for edge in edges:
            R_initial = tipping_point_rotations[edge].dot(self.obj.T_obj_world)

            # before the object settles
            initial_rotated_mesh = self.mesh.copy().apply_transform(R_initial.matrix)
            lowest_z = np.min(initial_rotated_mesh.vertices[:,2])
            # print lowest_z, edge_point1[2]
            # if lowest_z >= self.lowest_z: # object would topple
            if True:
                resting_pose = self.obj.obj.resting_pose(R_initial)
                final_poses.append(resting_pose)
                # final_poses.append(R_initial)
            else:
                print 'lowest z', lowest_z, self.lowest_z
                # object would rotate back onto original stable pose
                # (assuming finger doesn't keep pushing)
                final_poses.append(current_pose)
        return final_poses

    def combine_equivalent_poses(self, final_poses, vertex_probs):
        """
        Combines the probabilities of poses which are equivalent
        except for rotations along the z axis or translations

        Parameters
        ----------
        final_poses : :obj:`list` of :obj:`StablePose`
            list of poses the object ends up in after being toppled over each edge of size m
        vertex_probs : nxm :obj:`numpy.ndarray`
            probability of object landing in each of the m poses for each of the n topple actions

        Returns
        -------
        :obj:`list` of :obj:`StablePose` of size o
            list of unique Stable Poses
        nxo :obj:`numpy.ndarray`
            probability of object landing in each of the o unique stable poses
            for each of the n topple actions
        """
        i = 0
        edge_inds = list(np.arange(vertex_probs.shape[1]))
        grouped_poses = []
        grouped_edges = []
        while i < len(edge_inds):
            equivalent_edges = [edge_inds[i]]
            curr_edge_ind = edge_inds[i]
            j = i+1
            while j < len(edge_inds):
                if is_equivalent_pose(self.final_poses[curr_edge_ind], self.final_poses[edge_inds[j]]):
                    equivalent_edges.append(edge_inds[j])
                    edge_inds.pop(j)
                else:
                    j += 1
            grouped_poses.append(self.final_poses[curr_edge_ind])
            grouped_edges.append(equivalent_edges)
            i += 1
        vertex_probs = \
            np.hstack([np.sum(vertex_probs[:,edges], axis=1).reshape(-1,1) for edges in grouped_edges])
        return grouped_poses, vertex_probs

    def finger_slips(self, push_direction, normal, finger_friction_noise, tipping_point_rotation):
        def finger_slip_helper(normal):
            parallel_component = push_direction.dot(-normal)
            #perpend_component = np.linalg.norm(push_direction - normal*parallel_component)
            perpend_component = math.sqrt(1-min(parallel_component**2,1))
            # Finger slips before even starting a rotation
            #return parallel_component / (perpend_component + 1e-5) <= 1/(finger_friction_noise * self.finger_friction_coeff)
            return perpend_component / (parallel_component + 1e-5) >= finger_friction_noise * self.finger_friction_coeff

        if finger_slip_helper(normal):
            return True
        new_normal = (tipping_point_rotation * Point(normal, 'world')).data
        if finger_slip_helper(new_normal):
            return True
        return False
        
    def predict(self, vertices, normals, push_directions, use_sensitivity=True):
        """
        Predict distribution of poses if pushed at vertices
        
        Parameters
        ----------
        vertices : nx3 :obj:`numpy.ndarray`
        push_directions : nx3 :obj:`numpy.ndarray`
        use_sensitivity : bool
        
        Returns
        -------
        :obj:`list` of :obj:`RigidTransform`
        :obj:`list` of float
        """
        n_trials = 1
        if use_sensitivity:
            n_trials = self.n_trials
            (
                vertices, 
                normals, 
                push_directions, 
                ground_friction_noises, 
                finger_friction_noises
            )  = self.add_noise(
                vertices, 
                normals, 
                push_directions, 
                n_trials
            )

        tipping_point_rotations = self.tipping_point_rotations()
                    
        # Check whether each push point topples or not
        vertex_probs = [] # probability of toppling over each edge
        # number of predicted times it will topple for the current vertex
        current_vertex_counts = np.zeros(len(self.edge_points))
        i = 0
        a = time()
        min_required_forces = []
        while i < len(vertices):
            vertex = vertices[i]
            normal = normals[i]
            push_direction = push_directions[i]
            finger_friction_noise = finger_friction_noises[i] if use_sensitivity else 1
            ground_friction_noise = ground_friction_noises[i] if use_sensitivity else 1
            vertex_forces = []
            
            # elif vertex[2] < self.thresh or i % n_trials == int(n_trials * fraction_before_short_circuit):
            #     i += 1
            #     continue
            
            # short circuit the trial if the vertex is too low or the noise caused the 
            # finger to miss the object 
            if not np.array_equal(normal, np.array([0,0,0])):
                # Go through each edge and find out which one requires the least force to topple
                # from the vertex
                min_required_force = np.inf # required force to topple over topple_edge
                topple_edge = None # current edge we think it will topple over
                for (
                    curr_edge,
                    (edge_point1, edge_point2), 
                    com_projected_on_edge,
                    max_moment, 
                    max_tangential_force,
                    tipping_point_rotation
                ) in zip(
                    range(len(self.edge_points)),
                    self.edge_points, 
                    self.com_projected_on_edges, 
                    self.max_moments, 
                    self.max_tangential_forces,
                    tipping_point_rotations
                ):
                    # Finding if finger slips on object (Condition 1)
                    if self.finger_slips(push_direction, normal, finger_friction_noise, tipping_point_rotation):
                        continue

                    # finding required force to topple
                    required_force = self.required_force(
                        vertex, 
                        push_direction, 
                        edge_point1, 
                        edge_point2, 
                        com_projected_on_edge
                    )
                    if required_force < 0 or required_force >= min_required_force:
                        continue
                    f_x, f_y, f_z = required_force * push_direction
                        
                    induced_torque = self.induced_torque(
                        vertex, 
                        push_direction, 
                        edge_point1, 
                        edge_point2, 
                        com_projected_on_edge, 
                        required_force
                    )
                    finger_friction_moment = self.finger_friction_moment(f_z, edge_point1, edge_point2)

                    if ( # Condition 2
                        (f_x / (ground_friction_noise * max_tangential_force + self.ground_friction_coeff * f_z+1e-5))**2 + 
                        (f_y / (ground_friction_noise * max_tangential_force + self.ground_friction_coeff * f_z+1e-5))**2 + 
                        (induced_torque / (ground_friction_noise * max_moment + finger_friction_moment))**2 < 1
                    ):
                        min_required_force = required_force
                        topple_edge = curr_edge
                vertex_forces.append(min(min_required_force, self.max_force))
                if topple_edge is not None: # if it topples over at least one edge
                    current_vertex_counts[topple_edge] += 1

            i += 1
            if i % n_trials == int(n_trials * self.fraction_before_short_circuit) \
                and np.sum(current_vertex_counts) == 0:
                i = math.ceil(i / float(n_trials)) * n_trials
                vertex_forces.extend([self.max_force] * (n_trials - len(vertex_forces)))
            # If we have gone through each noisy sample at the current vertex, record this vertex, 
            # and clear counts for next vertex
            if i % n_trials == 0:
                vertex_probs.append(current_vertex_counts / n_trials)
                current_vertex_counts = np.zeros(len(self.edge_points))
                min_required_forces.append(np.mean(vertex_forces))
        vertex_probs = np.array(vertex_probs)
        if self.log:
            print 'probability time', time() - a

        # Only calculate the final pose for edges that are actually toppled over
        a = time()
        non_zero_edges = np.arange(len(self.edge_points))[np.sum(vertex_probs, axis=0) != 0]
        vertex_probs = vertex_probs[:,non_zero_edges]
        self.final_poses = self.map_edge_to_pose(non_zero_edges, tipping_point_rotations)
        if self.log:
            print 'pose time', time() - a

        # adding the probability of not toppling
        not_topple_prob = 1 - np.sum(vertex_probs, axis=1, keepdims=True)
        vertex_probs = np.hstack([not_topple_prob, vertex_probs])
        self.final_poses.insert(0, self.obj.T_obj_world)
        self.vertices = vertices
        self.push_directions = push_directions
        
        grouped_poses, vertex_probs = self.combine_equivalent_poses(self.final_poses, vertex_probs)
        return grouped_poses, vertex_probs, np.array(min_required_forces)
        #return self.final_poses, vertex_probs, np.array(min_required_forces)

class TopplingDatasetModel():
    def __init__(self, datasets):
        self.datasets = {dataset_name:TensorDataset.open(datasets[dataset_name]) for dataset_name in datasets.keys()}
        self.obj_ids, self.id_to_datapoint = {}, []
        for dataset_name in datasets.keys():
            with open(datasets[dataset_name]+"/obj_ids.json", "r") as read_file:
                obj_ids = json.load(read_file)
            dataset = self.datasets[dataset_name]
            for i in range(dataset.num_datapoints):
                datapoint = dataset.datapoint(i)
                self.id_to_datapoint.append((datapoint['obj_id'], i))
            self.obj_ids[dataset_name] = obj_ids
        self.id_to_datapoint = {k: list(v) for k, v in groupby(self.id_to_datapoint, key=lambda x: x[0])} 
        
    def load_object(self, state):
        self.obj = state.obj
        self.com = self.obj.mesh.center_mass

        obj_dataset_name = self.obj.key.split('~')[0]
        dataset = self.datasets[obj_dataset_name]
        obj_ids = self.obj_ids[obj_dataset_name]
        self.datapoint = None

        similarity = np.inf
        # for i in range(dataset.num_datapoints):
        # for i in [101]:
        obj_specific_datapoints = self.id_to_datapoint[obj_ids[self.obj.key]]
        for _, i in obj_specific_datapoints:
            datapoint = dataset.datapoint(i)
            if obj_ids[self.obj.key] != datapoint['obj_id']:
               continue

            rot, trans = RigidTransform.rotation_and_translation_from_matrix(datapoint['obj_pose'])
            obj_pose = RigidTransform(rot, trans, 'obj', 'world')
            pose_diff_val = pose_diff(obj_pose, self.obj.T_obj_world)
            if pose_diff_val < similarity:
                self.datapoint = datapoint
                similarity = pose_diff_val
        if self.datapoint == None:
            raise ValueError('Object {} not in database'.format(self.obj.key))

    def predict(self):
        rot, trans = RigidTransform.rotation_and_translation_from_matrix(self.datapoint['obj_pose'])
        dataset_pose = RigidTransform(rot, trans, 'obj', 'world')
        dataset_final_poses = self.datapoint['final_poses']
        num_vertices = self.datapoint['num_vertices']
        final_poses = []
        for i in range(20):
            final_pose_mat = dataset_final_poses[i*4:(i+1)*4]
            if np.array_equal(final_pose_mat, np.zeros((4,4))):
                break
            rot, trans = RigidTransform.rotation_and_translation_from_matrix(final_pose_mat)
            final_poses.append(RigidTransform(rot, trans, 'obj', 'world'))
        vertices = self.datapoint['vertices'][:num_vertices]
        normals = self.datapoint['normals'][:num_vertices]
        vertices = (self.obj.T_obj_world * dataset_pose.inverse() * PointCloud(vertices.T, 'world')).data.T
        normals = (self.obj.T_obj_world * dataset_pose.inverse() * PointCloud(normals.T, 'world')).data.T


        vertex_probs = self.datapoint['vertex_probs'][:num_vertices, :len(final_poses)]
        required_forces = self.datapoint['min_required_forces'][:num_vertices]

        return vertices, normals, final_poses, vertex_probs, required_forces
