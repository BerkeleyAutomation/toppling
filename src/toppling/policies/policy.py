import numpy as np
from copy import deepcopy
from scipy.spatial import ConvexHull
import logging

from trimesh import sample 
from dexnet.envs import MultiEnvPolicy

def normalize(vec, axis=None):
    return vec / np.linalg.norm(vec) if axis == None else vec / np.linalg.norm(vec, axis=axis).reshape((-1,1))

class TopplingPolicy(MultiEnvPolicy):
    def __init__(self):
        MultiEnvPolicy.__init__(self)
        #self.obj_friction_coeff = config['obj_friction_coeff']
        #self.ground_friction_coeff = config['ground_friction_coeff']
        self.ground_friction_coeff = .5
    
    def predict(self, mesh, direction):
        def positive_rotation(force, force_origin, point):
            up = np.array([0,0,1])
            logging.info(('shapes force', force.shape, force_origin.shape, point.shape))
            return np.arccos(force.dot(up)) > np.arccos((point - force_origin).dot(up))
            
        logging.info('here')
        obj = copy(state.objs[0])
        cvh_mesh = obj.mesh.convex_hull
        #com = obj.T_obj_world.matrix.dot(np.append(obj.mesh.center_mass, [1]))
        com = np.concatenate((obj.T_obj_world.translation, [1]))

        # Homogeneous coordinates
        n = cvh_mesh.triangles.shape[0]
        triangles = np.concatenate([copy(cvh_mesh.triangles), np.ones((n, 3, 1))], axis=-1)
        triangles_world = np.einsum('ij,ndj->ndi', obj.T_obj_world.matrix, triangles)
        
        #z_components = triangles_world[:,:,2]
        #z_components = np.mean(z_components, axis=1)
        #bottom_triangles_world = triangles_world[z_components < 1.1 * np.min(z_components)]
        #bottom_points = bottom_triangles_world.reshape(-1, 4)
        #direction = np.append(direction / np.linalg.norm(direction), [1])
        #furthest_vertex_ind = np.argmax(bottom_points.dot(direction))
        #furthest_vertex = bottom_points[furthest_vertex_ind]

        # Find push vertex
        n = obj.mesh.vertices.shape[0]
        vertices = np.concatenate([copy(obj.mesh.vertices), np.ones((n,1))], axis=-1)
        vertices_world = np.einsum('ij,nj->ni', obj.T_obj_world.matrix, vertices)[:,:3]
        projected_vectors = (vertices_world - com[:3])[:,:2]
        projected_vectors = projected_vectors / np.linalg.norm(projected_vectors, axis=1)[:, None]
        projected_angles = np.arccos(projected_vectors.dot(-direction[:2]))
        #vertex_height_indices = np.argsort(vertices_world[:,2])
        subset_ind = np.argmax(vertices_world[projected_angles < np.pi/9, 2]) #Finds the index of the highest vertex in the range behind the COM
        push_vertex_ind = np.arange(n)[projected_angles < np.pi/9][subset_ind] #Finds the index in the list of all vertices

        #Find toppling rotation center
        z_components = vertices_world[:,2]
        thresh = .99 * np.min(z_components) + .01 * np.max(z_components)
        bottom_vertices = vertices_world[z_components < thresh]
        direction = direction / np.linalg.norm(direction)
        furthest_vertex_ind = np.argmax(bottom_vertices.dot(direction))
        furthest_vertex = bottom_vertices[furthest_vertex_ind]
       
        # Check if toppling is possible
        # https://pdfs.semanticscholar.org/3383/6148d1f5bccda4c86750e381181f750ef814.pdf
        p2 = furthest_vertex[:3]
        p1 = copy(com[:3])
        p1[2] = p2[2] + np.arctan(np.pi/2 - self.ground_friction_coeff)
        push_surface_normal = np.concatenate((obj.mesh.vertex_normals[push_vertex_ind], [1]))
        #if not positive_rotation(obj.T_obj_world.matrix.dot(push_surface_normal)[:3], vertices_world[push_vertex_ind], p1):
        #    return None, None, None
        
        # Defining axis of rotation and how much to rotate
        y = - np.cross(direction[:3], np.array([0,0,1]))
        y = y / np.linalg.norm(y)
        logging.info(('furthest, com', furthest_vertex, com))
        tmpx = furthest_vertex - com[:3] - y.dot(furthest_vertex - com[:3]) * y #furthest vertex - projection of furthest_vertex onto y
        logging.info(('tmpx', tmpx))
        x = tmpx / np.linalg.norm(tmpx)
        logging.info(('x', x))
        topple_angle = math.acos(np.dot(x, np.array([0,0,-1]))) + 1e-2
        logging.info(('angle', topple_angle))

        # Rodrigues Formula
        #y_hat = np.array([[0, -y[2], y[1]], [y[2], 0, -y[0]], [-y[1], y[0], 0]])
        #R = RigidTransform(np.eye(3) + np.sin(topple_angle) * y_hat + (1 - np.cos(topple_angle)) * y_hat.dot(y_hat), from_frame='world')
        
        #R = RigidTransform(translation=furthest_vertex[:3], from_frame='world').dot(R).dot(RigidTransform(translation=-furthest_vertex[:3], from_frame='world')).dot(obj.T_obj_world)
        tmpR = RigidTransform.rotation_from_axis_and_origin(y[:3], furthest_vertex[:3], topple_angle).dot(obj.T_obj_world)
        R = obj.resting_pose(tmpR) 
        #return R, vertices_world[push_vertex_ind], vertices_world[projected_angles < np.pi/9]
        #return R, vertices_world[push_vertex_ind], np.array([furthest_vertex[:3], com[:3] + tmpx])
        push_vertex = vertices_world[push_vertex_ind]
        #return R, [push_vertex, push_vertex + .03*push_surface_normal[:3]], [], tmpR
        #return R, [], [[furthest_vertex[:3] - .02*y, furthest_vertex[:3] + .02*y]], tmpR
        return R, [], [], tmpR

    def predict_pose(self, mesh, edge_point1, edge_point2):
        pass

    def will_topple(self, obj, push_vertex, push_direction):
        def predict_topple(
            edge_point1, 
            edge_point2,
            use_sensitivity=True
        ):
            v = com - edge_point1
            s = normalize(edge_point2 - edge_point1)
            logging.info(('s_shape', v.shape, s.shape))
            com_projected_on_edge = v.dot(s) * s + edge_point1
            push_direction = com_projected_on_edge - com
            push_direction[2] = 0
            push_direction = normalize(push_direction)
            
            # finding maximum force and torque the friction can # max torque increase due to the finger pressing down on the objectresist (approximated by inscribed rectangle)
            offset_dist = np.linalg.norm(edge_point1 - edge_point2)
            offsets = np.linspace(0, offset_dist, num_approx)
            offsets_relative_to_com = offsets - np.linalg.norm(com_projected_on_edge - edge_point1)
            #paths = obj.mesh.section_multiplane(edge_point1, s, offsets)
            #mass_per_unit = np.array([0 if paths[index] == None else paths[index].area for index in range(num_approx)])
            mass_per_unit = np.array([mass / float(num_approx)] * num_approx) # Using a uniform pressure distribution along the contact edge
            pressure_dist = mass_per_unit * mass * 9.8 / (np.sum(mass_per_unit) + offset_dist / num_approx) # From mass and from the finger pressing down on the object 
            max_moment = self.ground_friction_coeff * pressure_dist.dot(np.abs(offsets_relative_to_com))
            #max_moment = self.ground_friction_coeff * mass * 9.8 * (y0**2 + y1**2) / 2
            max_tangential_force = self.ground_friction_coeff * (mass * 9.8) # mu F_n
            logging.info(('max', max_moment, max_tangential_force))


            # sample push points
            vertices, face_ind = sample.sample_surface_even(mesh, 1000)
            projected_vectors = (vertices - com)[:,:2]
            projected_vectors = projected_vectors / np.linalg.norm(projected_vectors, axis=1)[:, None]
            projected_angles = np.arccos(projected_vectors.dot(-push_direction[:2]))
            vertices = vertices[projected_angles < max_push_angle]
            normals = mesh.face_normals[face_ind[projected_angles < max_push_angle]]

            n_trials = 1
            if use_sensitivity:
                n_trials = 25
                sigma = .00125 # noise for finger position
                vertices = np.repeat(vertices, n_trials, axis=0)
                vertices = vertices + np.random.normal(scale=sigma, size=vertices.shape)
                push_directions = np.repeat(-normals, n_trials, axis=0)
                intersect_loc, _, face_ind = mesh.ray.intersects_location(vertices, normals, multiple_hits=False)
                # Finding the normal of the noisy point
                logging.info(('shape', intersect_loc.shape))
                closest_intersections = np.argmin(np.linalg.norm(intersect_loc - vertices, axis=1))
#                vertices = 
#                normals = 

                noises = 1 + np.random.normal(scale=.1, size=vertices.shape[0]) / self.ground_friction_coeff

            # Check whether each push point topples or not
            fraction_before_short_circuit = .4
            successes, required_forces = [], []
            for i in range(len(vertices)):
                if i % n_trials == 0:
                    current_num_successes = 0
                if i % n_trials == int(n_trials * fraction_before_short_circuit):
                    num_left = int(n_trials * (1-fraction_before_short_circuit))
                    successes.extend([False] * num_left)
                    required_forces.extend([0] * num_left)
                    i += num_left
                    
                    
                vertex = vertices[i]
                normal = normals[i]
                push_direction = push_directions[i]
                noise = noises[i] if use_sensitivity else 1
                    

#                if vertex[2] < thresh:
#                    continue
                # finding required force to topple
                vertex_projected_on_edge = (vertex - edge_point1).dot(s)*s + edge_point1 
                f_max = normalize(np.cross(edge_point1 - vertex, s))
                r_f = np.linalg.norm(vertex - vertex_projected_on_edge)
                cos_push_vertex_edge_angle = push_direction.dot(f_max)
                g_max = normalize(-np.cross(edge_point1 - com, s))
                r_g = np.linalg.norm(com - com_projected_on_edge)
                cos_com_edge_angle = (-up).dot(g_max)
                vertex_projected_on_edge = (vertex - edge_point1).dot(s) + edge_point1
                
                required_force = mass * 9.8 * cos_com_edge_angle * r_g / (cos_push_vertex_edge_angle * r_f + 1e-5)
                if required_force < 0:
                    continue
                f_x = required_force * push_direction[0]
                f_y = required_force * push_direction[1]
                f_z = required_force * push_direction[2]
                    
                # finding if required toppling force can be resisted by friction
                r = vertex - com_projected_on_edge
                r[2] = 0
                max_z_torque_dir = normalize(np.cross(-r, up))
                cos_push_vertex_z_angle = push_direction.dot(max_z_torque_dir)
                r = np.linalg.norm(r)
                induced_torque = required_force * cos_push_vertex_z_angle * r
                #induced_torque = required_force * np.linalg.norm(np.cross(r, push_direction))
                # max torque increase due to the finger pressing down on the object
                #finger_friction_moment = (f_z / np.sum(pressure_mask)) * pressure_mask.astype(int).dot(offsets_relative_to_com) 
                downward_force_dist = np.array([f_z / float(num_approx)] * num_approx)
                finger_friction_moment = self.ground_friction_coeff * downward_force_dist.dot(np.abs(offsets_relative_to_com))

                # Finding if finger slips on object
                parallel_component = push_direction.dot(-normal)
                perpend_component = np.linalg.norm(push_direction + normal*parallel_component)
                
                if (
                    (f_x / (noise * max_tangential_force + self.ground_friction_coeff * f_z))**2 + 
                    (f_y / (noise * max_tangential_force + self.ground_friction_coeff * f_z))**2 + 
                    (induced_torque / (noise * max_moment + finger_friction_moment))**2 < 1
                ) and (
                    parallel_component / (perpend_component + 1e-5) > noise * self.ground_friction_coeff
                ):
                    successes.append(True)
                else:
                    successes.append(False)
                required_forces.append(required_force)
                
            probabilities = np.mean(np.array(successes).reshape((-1,num_trials)), axis=1)
            return vertices, normals, np.array(probabilities), np.array(required_forces)

        # potential problems: edge points, object not rotated when finding planes
        mesh = deepcopy(obj.mesh).apply_transform(obj.T_obj_world.matrix)
        com = mesh.center_mass
        up = np.array([0,0,1])
        num_approx = 30
        max_push_angle = np.pi/2
        mass = 1
       
        # Finding toppling edge
        z_components = np.around(mesh.vertices[:,2], 3)
        lowest_z = np.min(z_components)
        bottom_points = mesh.vertices[z_components == lowest_z][:,:2]
        bottom_points = bottom_points[ConvexHull(bottom_points).vertices]
        centered_bottom_points = bottom_points - com[:2]
        angles = np.arctan2(centered_bottom_points[:,1], centered_bottom_points[:,0]) # finding pair of points with maximal angle
        angles = angles - np.roll(angles,1) # angles of point - angles of next point
        angles = (angles + 2*np.pi) % (2*np.pi) # handling the fact that some angles are negative
        largest_diff_index = np.argmax(angles)
        next_idx = (largest_diff_index - 1) % len(bottom_points)
        edge_point2 = np.append(bottom_points[largest_diff_index], [lowest_z]) #making point 3d again, except projected onto the bottom plane
        edge_point1 = np.append(bottom_points[next_idx], [lowest_z])

        vertices, normals, probabilities, required_forces = predict_topple(edge_point1, edge_point2)
        max_prob = np.max(probabilities)
        required_force = np.min(required_forces[probabilities == max_prob])
        min_force_idx = np.argmin(required_forces[probabilities == max_prob])
        vertex = vertices[probabilities == max_prob][min_force_idx]
        normal = normals[probabilities == max_prob][min_force_idx]

        return obj.T_obj_world, vertices, probabilities, np.array([edge_point1, edge_point2]), obj.T_obj_world, push_direction

    def will_topple_new(self, obj, push_vertex, push_direction):
        def predict_topple(
            edge_point1, 
            edge_point2,
            use_sensitivity=True,
            vertices=None,
            normals=None
        ):
            v = com - edge_point1
            s = normalize(edge_point2 - edge_point1)
            com_projected_on_edge = v.dot(s) * s + edge_point1
            push_direction = com_projected_on_edge - com
            push_direction[2] = 0
            push_direction = normalize(push_direction)
            
            # finding maximum force and torque the friction can # max torque increase due to the finger pressing down on the objectresist (approximated by inscribed rectangle)
            offset_dist = np.linalg.norm(edge_point1 - edge_point2)
            offsets = np.linspace(0, offset_dist, num_approx)
            offsets_relative_to_com = offsets - np.linalg.norm(com_projected_on_edge - edge_point1)
            #paths = obj.mesh.section_multiplane(edge_point1, s, offsets)
            #mass_per_unit = np.array([0 if paths[index] == None else paths[index].area for index in range(num_approx)])
            mass_per_unit = np.array([mass / float(num_approx)] * num_approx) # Using a uniform pressure distribution along the contact edge
            pressure_dist = mass_per_unit * mass * 9.8 / (np.sum(mass_per_unit) + offset_dist / num_approx) # From mass and from the finger pressing down on the object 
            max_moment = self.ground_friction_coeff * pressure_dist.dot(np.abs(offsets_relative_to_com))
            #max_moment = self.ground_friction_coeff * mass * 9.8 * (y0**2 + y1**2) / 2
            max_tangential_force = self.ground_friction_coeff * (mass * 9.8) # mu F_n

            if vertices is None:
                # sample push points
                vertices, face_ind = sample.sample_surface_even(mesh, 1000)
                # vertices = mesh.vertices
                projected_vectors = (vertices - com)[:,:2]
                projected_vectors = projected_vectors / np.linalg.norm(projected_vectors, axis=1)[:, None]
                projected_angles = np.arccos(projected_vectors.dot(-push_direction[:2]))
                vertices = vertices[projected_angles < max_push_angle]
                # normals = mesh.vertex_normals[projected_angles < max_push_angle]
                normals = mesh.face_normals[face_ind[projected_angles < max_push_angle]]

            # Check whether each push point topples or not
            probabilities, required_forces = [], []
            n_trials = 25 if use_sensitivity else 1
            #sigma = (np.max(z_components) - np.min(z_components)) / 10 #.0025
            #logging.info(('sigma', sigma))
            sigma = .00125
            for i in range(len(vertices)):
                successes = 0
                required_force_sum = 0
                trial_num = 0
                while trial_num < n_trials:
                    trial_num += 1
                    if trial_num > 10 and successes == 0: # short circuit if we are reasonably confident all trials will be failures
                        trial_num = n_trials
                        continue
                    push_direction = -normals[i]
                    if use_sensitivity:
                        vertex = vertices[i] + np.random.normal(scale=sigma, size=3)
                        tmp_vertex = vertex
                        intersect_loc, _, face_ind = mesh.ray.intersects_location(
                            np.expand_dims(vertex + normals[i], axis=0), 
                            np.expand_dims(push_direction, axis=0)
                        )
                        
                        if len(intersect_loc) == 0: # noise causes intersection to miss the object
                            continue
                        closest_intersection = np.argmin(np.linalg.norm(intersect_loc - vertex, axis=1))
                        vertex = intersect_loc[closest_intersection]
                        normal = mesh.face_normals[face_ind[closest_intersection]]
                        noise = 1 + np.random.normal(scale=.1) / self.ground_friction_coeff
                    else:
                        vertex = vertices[i]
                        normal = normals[i]
                        noise = 1

                    # if vertex[2] < thresh:
                    #     continue
                    # finding required force to topple
                    vertex_projected_on_edge = (vertex - edge_point1).dot(s)*s + edge_point1 
                    f_max = normalize(np.cross(edge_point1 - vertex, s))
                    r_f = np.linalg.norm(vertex - vertex_projected_on_edge)
                    cos_push_vertex_edge_angle = push_direction.dot(f_max)
                    g_max = normalize(-np.cross(edge_point1 - com, s))
                    r_g = np.linalg.norm(com - com_projected_on_edge)
                    cos_com_edge_angle = (-up).dot(g_max)
                    vertex_projected_on_edge = (vertex - edge_point1).dot(s) + edge_point1
                    
                    required_force = mass * 9.8 * cos_com_edge_angle * r_g / (cos_push_vertex_edge_angle * r_f + 1e-5)
                    if required_force < 0:
                        continue
                    f_x = required_force * push_direction[0]
                    f_y = required_force * push_direction[1]
                    f_z = required_force * push_direction[2]
                    
                    # finding if required toppling force can be resisted by friction
                    r = vertex - com_projected_on_edge
                    r[2] = 0
                    max_z_torque_dir = normalize(np.cross(-r, up))
                    cos_push_vertex_z_angle = push_direction.dot(max_z_torque_dir)
                    r = np.linalg.norm(r)
                    induced_torque = required_force * cos_push_vertex_z_angle * r
                    #induced_torque = required_force * np.linalg.norm(np.cross(r, push_direction))
                    # max torque increase due to the finger pressing down on the object
                    #finger_friction_moment = (f_z / np.sum(pressure_mask)) * pressure_mask.astype(int).dot(offsets_relative_to_com) 
                    downward_force_dist = np.array([f_z / float(num_approx)] * num_approx)
                    finger_friction_moment = self.ground_friction_coeff * downward_force_dist.dot(np.abs(offsets_relative_to_com))

                    # Finding if finger slips on object
                    parallel_component = push_direction.dot(-normal)
                    perpend_component = np.linalg.norm(push_direction + normal*parallel_component)
                    
                    if (
                        (f_x / (noise * max_tangential_force + self.ground_friction_coeff * f_z))**2 + 
                        (f_y / (noise * max_tangential_force + self.ground_friction_coeff * f_z))**2 + 
                        (induced_torque / (noise * max_moment + finger_friction_moment))**2 < 1
                    ) and (
                        parallel_component / (perpend_component + 1e-5) > noise * self.ground_friction_coeff
                    ):
                        successes += 1
                    required_force_sum += required_force
                probabilities.append(float(successes) / n_trials)
                required_forces.append(float(required_force_sum) / n_trials)
            
            return vertices, normals, np.array(probabilities), np.array(required_forces)

        # potential problems: edge points, object not rotated when finding planes
        mesh = deepcopy(obj.mesh).apply_transform(obj.T_obj_world.matrix)
        com = mesh.center_mass
        up = np.array([0,0,1])
        num_approx = 30
        mass = 1
       
        # Finding toppling edge
        z_components = np.around(mesh.vertices[:,2], 3) #rounding to 3 decimal places
        lowest_z = np.min(z_components)
        bottom_points = mesh.vertices[z_components == lowest_z]
        bottom_points = bottom_points[ConvexHull(bottom_points).vertices]
        num_bottom = bottom_points.shape[0]
        logging.info(('num_bottom', num_bottom))

        vertices, face_ind = sample.sample_surface_even(mesh, 1000)
        normals = mesh.face_normals[face_ind]
        
        bottom_points_rolled = np.roll(bottom_points, 1, axis=0)
        vs = -bottom_points_rolled + com
        ss = normalize(bottom_points - bottom_points_rolled, axis=1)
        com_projected_on_edges = np.multiply(np.einsum('ij,ij->i', vs, ss).reshape((-1,1)), ss) + bottom_points
        
        max_tangential_force = self.ground_friction_coeff * (mass * 9.8) # mu F_n
        edge_lengths = np.linalg.norm(bottom_points - bottom_points_rolled, axis=1)
        dist_from_com = np.linalg.norm(bottom_points_rolled - com_projected_on_edges, axis=1)
        offsets_relative_to_com = [np.linspace(-dist_from_com[i], edge_lengths[i] - dist_from_com[i], num_approx) for i in range(num_bottom)]
        mass_per_unit = np.array([mass / float(num_approx)] * num_approx) # Using a uniform pressure distribution along the contact edge
        pressure_dists = [mass_per_unit * mass * 9.8 / (np.sum(mass_per_unit) + edge_len / num_approx) for edge_len in edge_lengths]
        max_moments = [self.ground_friction_coeff * pressure_dists[i].dot(np.abs(offsets_relative_to_com[i])) for i in range(num_bottom)]
        
#        logging.info(('shapes', vertices.shape, bottom_points_rolled.shape, ss.shape))
#        vertices_projected_on_edges = np.einsum('ij,kj->ik', vertices, -bottom_points_rolled).dot(ss) # i,k: i is vertex, j is edge
#        logging.info((vertices_projected_on_edges))
#        sys.exit()
        vertices_to_edges = np.array([vertices[i] - bottom_points_rolled for i in range(len(vertices))])
        logging.info(('shapes', vertices_to_edges.shape, ss.shape))
        vertices_projected_on_edges = vertices_to_edges.dot(ss.T)
        vertices_projected_on_edges
        # Computing required force
        vertices_projected_on_edges = []
        for vertex in vertices:
            single_vertex_projections = []
            for i in range(num_bottom):
                single_vertex_projections.append((vertex - bottom_points_rolled[i]).dot(ss[i])*ss[i] + bottom_points_rolled[i])
            vertices_projected_on_edges.append(single_vertex_projections)
        vertices_projected_on_edges = np.array(vertices_projected_on_edges) # v, e, 3: v vertices, e edges, 3 elements per point
        f_max = normalize(np.cross(edge_point1 - vertex, ss))



        vertices, normals, probabilities, required_forces = predict_topple(edge_point1, edge_point2)
        max_prob = np.max(probabilities)
        required_force = np.min(required_forces[probabilities == max_prob])
        min_force_idx = np.argmin(required_forces[probabilities == max_prob])
        vertex = vertices[probabilities == max_prob][min_force_idx]
        normal = normals[probabilities == max_prob][min_force_idx]

        # Check if it topples on the edges next to it
#        right_idx = (largest_diff_index - 2) % len(bottom_points)
#        right_edge_point = np.append(bottom_points[right_idx], [lowest_z])
#        left_idx = (largest_diff_index + 1) % len(bottom_points)
#        left_edge_point = np.append(bottom_points[left_idx], [lowest_z])
#        _, _, probs_left, required_forces_left = predict_topple(edge_point2, left_edge_point, use_sensitivity=False, vertices=np.array([vertex]), normals=np.array([normal]))
#        _, _, probs_right, required_forces_right = predict_topple(right_edge_point, edge_point1, use_sensitivity=False, vertices=np.array([vertex]), normals=np.array([normal]))
#        logging.info(('left, right', probs_left, probs_right, required_forces_left, required_forces_right, required_force))


        # vertices = np.array([vertex, edge_point1, edge_point2, left_edge_point, right_edge_point])
        # probabilities = np.array([max_prob, 0, 0, 0.5, 0.5])
        # vertices = np.array([edge_point1, edge_point2])
        # probabilities = np.array([0,0])

        #return obj.T_obj_world, [np.array(not_topple), np.array(topple), np.array([edge_point1, edge_point2])], [], obj.T_obj_world, push_direction
        return obj.T_obj_world, vertices, probabilities, np.array([edge_point1, edge_point2]), obj.T_obj_world, push_direction
        
    def action(self, state):
        theta = 2*np.pi*np.random.random()
        #direction = np.array([np.cos(theta),np.sin(theta),0])
        direction = np.array([1,0,0])
        # GRASPINGENV	        
	    # com = state.objs[0].T_obj_world.translation
        com = state.T_obj_world.translation

        #R, set_of_points, set_of_lines, tmpR = self.predict(state, direction) 
        #R, set_of_points, set_of_lines, tmpR, direction = self.will_topple(state.objs[0], np.array([0,0,0]), direction)
        # GRASPINGENV
        # R, vertices, probabilities, tmpR, direction = self.will_topple(state.objs[0], np.array([0,0,0]), direction)
        R, vertices, probabilities, edge_points, tmpR, direction = self.will_topple(state, np.array([0,0,0]), direction)

        #thetas = np.linspace(0,2*np.pi, 12)
        #for theta in thetas:
        #    direction = np.array([np.cos(theta), np.sin(theta), 0])
        #    R, tmpx, abc = self.predict(state, direction)
        #    if R is not None:
        #        break
        #    logging.info(('un-topplable direction: ', direction))
        #return {'start': com, 'end': com + .1 * direction, 'final_state': R, 'set_of_points': set_of_points, 'set_of_lines': set_of_lines, 'tmpR': tmpR}
        return {'start': com, 'end': com + .1 * direction, 'final_state': R, 'vertices': vertices, 'probabilities': probabilities, 'tmpR': tmpR, 'edge_points': edge_points}
        
