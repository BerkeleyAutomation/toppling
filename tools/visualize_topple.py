import argparse
import numpy as np
import os
import colorsys
import random
from copy import deepcopy

from autolab_core import Point, RigidTransform, YamlConfig, TensorDataset
from dexnet.constants import *
from dexnet.envs import GraspingEnv
from dexnet.visualization import DexNetVisualizer3D as vis3d
# from ambicore.visualization import Visualizer3D as vis3d
from toppling.policies import SingleTopplePolicy, MultiTopplePolicy
from toppling import is_equivalent_pose, camera_pose, normalize, up
import trimesh

SEED = 107
CAMERA_POSE = camera_pose()
red = [1,0,0]
blue = [0,0,1]
purple = [.5,0,.5]
teal = [0,1,1]
pastel_orange = np.array([248,184,139])/255.0
pastel_blue = np.array([178,206,254])/255.0
light_blue = np.array([0.67843137, 0.84705882, 0.90196078])

def vis_axes(origin, y_dir):
    y = [origin, origin + .0075*y_dir]
    z = [origin, origin + .0075*up]
    x = [origin, origin + .0075*np.cross(y_dir, up)]
    vis3d.plot3d(x, color=[1,0,0], tube_radius=.0006)
    vis3d.plot3d(y, color=[0,1,0], tube_radius=.0006)
    vis3d.plot3d(z, color=[0,0,1], tube_radius=.0006)

def display_or_save(filename):
    if args.save:
        vis3d.save_loop(filename, starting_camera_pose=CAMERA_POSE)
    else:
        vis3d.show(starting_camera_pose=CAMERA_POSE)

def dotted_line(start, end):
    r = np.arange(0,1,.004/np.linalg.norm(start-end))
    i = 0
    lines = []
    while i < len(r)-1:
        lines.append([r[i]*start+(1-r[i])*end, r[i+1]*start+(1-r[i+1])*end])
        i += 2
    return np.array(lines)

def figure_0():
    action = policy.action(env.state, env)
    env.render_3d_scene()
    bottom_points = policy.toppling_model.bottom_points
    vis3d.plot3d(bottom_points[:2], color=[0,0,0], tube_radius=.001)

    mesh = env.state.mesh.copy().apply_transform(env.state.T_obj_world.matrix)
    mesh.fix_normals()
    direction = normalize([0, -.04, 0])
    origin = mesh.center_mass + np.array([0,.04,.09])
    intersect, _, face_ind = mesh.ray.intersects_location([origin], [direction], multiple_hits=False)
    normal = mesh.face_normals[face_ind[0]]
    start_point = intersect[0] + .06*normal
    end_point = intersect[0]
    shaft_points = [start_point, end_point]
    h1 = np.array([[1,0,0],[0,0.7071,-0.7071],[0,0.7071,0.7071]]).dot(-normal)
    h2 = np.array([[1,0,0],[0,0.7071,0.7071],[0,-0.7071,0.7071]]).dot(-normal)
    head_points = [end_point - 0.02*h2, end_point, end_point - 0.02*h1]
    vis3d.plot3d(shaft_points, color=[1,0,0], tube_radius=.002)
    vis3d.plot3d(head_points, color=[1,0,0], tube_radius=.002)
    vis3d.points(Point(end_point), scale=.004, color=[0,0,0])
    hand_pose = RigidTransform(
        rotation=policy.get_hand_pose(start_point, end_point),
        translation=start_point,
        from_frame='grasp',
        to_frame='world'
    )
    gripper = env.gripper(action)
    vis3d.mesh(gripper.mesh, hand_pose * gripper.T_mesh_grasp, color=light_blue)
    vis3d.show(starting_camera_pose=CAMERA_POSE)

    env.state.obj.T_obj_world = policy.toppling_model.final_poses[1]
    action = policy.grasping_policy.action(env.state)
    print 'q:', action.q_value
    env.render_3d_scene()
    vis3d.gripper(env.gripper(action), action.grasp(env.gripper(action)), color=light_blue)
    vis3d.show(starting_camera_pose=CAMERA_POSE)

def figure_1():
    env.state.obj.T_obj_world.translation += np.array([-.01,-.05,.001])
    action = policy.action(env.state)
    env.render_3d_scene()
    bottom_points = action.metadata['bottom_points']
    vis3d.plot3d(bottom_points[:2], color=[0,0,0], tube_radius=.0005)
    vis3d.points(Point(bottom_points[0]), color=[0,0,0], scale=.001)
    vis3d.points(Point(bottom_points[1]), color=[0,0,0], scale=.001)
    y_dir = normalize(bottom_points[1] - bottom_points[0])
    origin = policy.toppling_model.com_projected_on_edges[0] - .005*y_dir
    vis_axes(origin, y_dir)

    #mesh = env.state.mesh.copy().apply_transform(env.state.T_obj_world.matrix)
    #mesh.fix_normals()
    #direction = normalize([-.03, -.07, 0])
    #intersect, _, face_ind = mesh.ray.intersects_location([[.02, -.005, .09]], [direction], multiple_hits=False)
    #normal = mesh.face_normals[face_ind[0]]
    #start_point = intersect[0] + .03*normal
    #end_point = intersect[0]
    #shaft_points = [start_point, end_point]
    #h1 = np.array([[0.7071,-0.7071,0],[0.7071,0.7071,0],[0,0,1]]).dot(-normal)
    #h2 = np.array([[0.7071,0.7071,0],[-0.7071,0.7071,0],[0,0,1]]).dot(-normal)
    #head_points = [end_point - 0.01*h2, end_point, end_point - 0.01*h1]
    #vis3d.plot3d(shaft_points, color=[1,0,0], tube_radius=.001)
    #vis3d.plot3d(head_points, color=[1,0,0], tube_radius=.001)
    #vis3d.points(Point(end_point), scale=.002, color=[0,0,0])

    # Center of Mass
    #start_point = env.state.T_obj_world.translation - .0025*y_dir - np.array([0,0,.005])
    start_point = env.state.T_obj_world.translation
    end_point = start_point - np.array([0, 0, .03])
    vis3d.points(Point(start_point), scale=.002, color=[0,0,0])
    shaft_points = [start_point, end_point]
    h1 = np.array([[1,0,0],[0,0.7071,-0.7071],[0,0.7071,0.7071]]).dot(-up)
    h2 = np.array([[1,0,0],[0,0.7071,0.7071],[0,-0.7071,0.7071]]).dot(-up)
    head_points = [end_point - 0.01*h2, end_point, end_point - 0.01*h1]
    vis3d.plot3d(shaft_points, color=[1,0,0], tube_radius=.001)
    vis3d.plot3d(head_points, color=[1,0,0], tube_radius=.001)
    vis3d.show(starting_camera_pose=CAMERA_POSE)

def figure_2():
    env.state.obj.T_obj_world.translation += np.array([-.01,-.05,.001])
    action = policy.action(env.state)
    env.render_3d_scene()
    bottom_points = action.metadata['bottom_points']
    vis3d.plot3d(bottom_points[:2], color=[0,0,0], tube_radius=.0005)
    vis3d.points(Point(bottom_points[0]), color=[0,0,0], scale=.001)
    vis3d.points(Point(bottom_points[1]), color=[0,0,0], scale=.001)
    y_dir = normalize(bottom_points[1] - bottom_points[0])
    origin = policy.toppling_model.com_projected_on_edges[0] - .0025*y_dir
    
    mesh = env.state.mesh.copy().apply_transform(env.state.T_obj_world.matrix)
    mesh.fix_normals()
    direction = normalize([-.03, -.07, 0])
    intersect, _, face_ind = mesh.ray.intersects_location([[.02, -.005, .09]], [direction], multiple_hits=False)
    normal = mesh.face_normals[face_ind[0]]
    start_point = intersect[0] + .03*normal
    end_point = intersect[0]
    shaft_points = [start_point, end_point]
    h1 = np.array([[0.7071,-0.7071,0],[0.7071,0.7071,0],[0,0,1]]).dot(-normal)
    h2 = np.array([[0.7071,0.7071,0],[-0.7071,0.7071,0],[0,0,1]]).dot(-normal)
    head_points = [end_point - 0.01*h2, end_point, end_point - 0.01*h1]
    vis3d.plot3d(shaft_points, color=red, tube_radius=.001)
    vis3d.plot3d(head_points, color=red, tube_radius=.001)
    vis3d.points(Point(end_point), scale=.002, color=[0,0,0])

    # Center of Mass
    start_point = env.state.T_obj_world.translation - .0025*y_dir - np.array([0,0,.005])
    end_point = start_point - np.array([0, 0, .03])
    vis3d.points(Point(start_point), scale=.002, color=[0,0,0])
    shaft_points = [start_point, end_point]
    h1 = np.array([[1,0,0],[0,0.7071,-0.7071],[0,0.7071,0.7071]]).dot(-up)
    h2 = np.array([[1,0,0],[0,0.7071,0.7071],[0,-0.7071,0.7071]]).dot(-up)
    head_points = [end_point - 0.01*h2, end_point, end_point - 0.01*h1]
    vis3d.plot3d(shaft_points, color=red, tube_radius=.001)
    vis3d.plot3d(head_points, color=red, tube_radius=.001)

    # Dotted lines
    r_gs = dotted_line(start_point, origin)
    for r_g in r_gs:
        vis3d.plot3d(r_g, color=blue, tube_radius=.0006)
    s = normalize(bottom_points[1] - bottom_points[0])
    vertex_projected_on_edge = (intersect[0] - bottom_points[0]).dot(s)*s + bottom_points[0]
    r_fs = dotted_line(intersect[0], vertex_projected_on_edge)
    for r_f in r_fs:
        vis3d.plot3d(r_f, color=blue, tube_radius=.0006)
    vis3d.show(starting_camera_pose=CAMERA_POSE)

def figure_3():
    env.state.obj.T_obj_world.translation += np.array([-.01,-.05,.01])
    action = policy.action(env.state, env)
    mesh = env.state.obj.mesh.copy().apply_transform(env.state.T_obj_world.matrix)
    mesh = mesh.slice_plane([0,0,.0105], -up)
    from dexnet.grasping import GraspableObject3D
    env.state.obj = GraspableObject3D(mesh)
    env.render_3d_scene()
    bottom_points = action.metadata['bottom_points']
    vis3d.plot3d(bottom_points[:2], color=[0,0,0], tube_radius=.0005)
    vis3d.points(Point(bottom_points[0]), color=[0,0,0], scale=.001)
    vis3d.points(Point(bottom_points[1]), color=[0,0,0], scale=.001)
    y_dir = normalize(bottom_points[1] - bottom_points[0])
    origin = policy.toppling_model.com_projected_on_edges[0] - .0025*y_dir
    vis3d.points(Point(origin), color=[0,0,1], scale=.001)

    # t = .002
    # x = np.cross(y_dir, up)
    # while t < np.linalg.norm(origin - bottom_points[0]):
    #     start_point = origin - t*y_dir
    #     end_point = start_point + .0075*x
    #     shaft_points = [start_point, end_point]
    #     h1 = np.array([[0.7071,-0.7071,0],[0.7071,0.7071,0],[0,0,1]]).dot(x)
    #     h2 = np.array([[0.7071,0.7071,0],[-0.7071,0.7071,0],[0,0,1]]).dot(x)
    #     head_points = [end_point - 0.001*h2, end_point, end_point - 0.001*h1]
    #     vis3d.plot3d(shaft_points, color=purple, tube_radius=.0002)
    #     vis3d.plot3d(head_points, color=purple, tube_radius=.0002)
    #     t += .002
    x = np.cross(y_dir, up)
    t = .01
    arrow_dir = x
    start_point = origin - t*y_dir
    end_point = start_point + .0075*arrow_dir
    shaft_points = [start_point, end_point]
    h1 = np.array([[0.7071,-0.7071,0],[0.7071,0.7071,0],[0,0,1]]).dot(x)
    h2 = np.array([[0.7071,0.7071,0],[-0.7071,0.7071,0],[0,0,1]]).dot(x)
    head_points = [end_point - 0.001*h2, end_point, end_point - 0.001*h1]
    vis3d.plot3d(shaft_points, color=purple, tube_radius=.0002)
    vis3d.plot3d(head_points, color=purple, tube_radius=.0002)
    
    # t = .000
    # while t < np.linalg.norm(origin - bottom_points[1]):
    #     arrow_dir = x
    #     start_point = origin + t*y_dir
    #     end_point = start_point + .0075*arrow_dir
    #     shaft_points = [start_point, end_point]
    #     h1 = np.array([[0.7071,-0.7071,0],[0.7071,0.7071,0],[0,0,1]]).dot(arrow_dir)
    #     h2 = np.array([[0.7071,0.7071,0],[-0.7071,0.7071,0],[0,0,1]]).dot(arrow_dir)
    #     head_points = [end_point - 0.001*h2, end_point, end_point - 0.001*h1]
    #     vis3d.plot3d(shaft_points, color=purple, tube_radius=.0002)
    #     vis3d.plot3d(head_points, color=purple, tube_radius=.0002)
    #     t += .002
    
    t = .004
    arrow_dir = x
    start_point = origin + t*y_dir
    end_point = start_point + .0075*arrow_dir
    shaft_points = [start_point, end_point]
    h1 = np.array([[0.7071,-0.7071,0],[0.7071,0.7071,0],[0,0,1]]).dot(arrow_dir)
    h2 = np.array([[0.7071,0.7071,0],[-0.7071,0.7071,0],[0,0,1]]).dot(arrow_dir)
    head_points = [end_point - 0.001*h2, end_point, end_point - 0.001*h1]
    vis3d.plot3d(shaft_points, color=purple, tube_radius=.0002)
    vis3d.plot3d(head_points, color=purple, tube_radius=.0002)

    #arrow_dir = np.cross(y_dir, up)
    #start_point = origin - .01*y_dir
    #end_point = start_point + .0075*arrow_dir
    #shaft_points = [start_point, end_point]
    #h1 = np.array([[0.7071,-0.7071,0],[0.7071,0.7071,0],[0,0,1]]).dot(arrow_dir)
    #h2 = np.array([[0.7071,0.7071,0],[-0.7071,0.7071,0],[0,0,1]]).dot(arrow_dir)
    #head_points = [end_point - 0.001*h2, end_point, end_point - 0.001*h1]
    #vis3d.plot3d(shaft_points, color=purple, tube_radius=.0002)
    #vis3d.plot3d(head_points, color=purple, tube_radius=.0002)

    #arrow_dir = -up
    #end_point = start_point + .0075*arrow_dir
    #shaft_points = [start_point, end_point]
    #h1 = np.array([[1,0,0],[0,0.7071,-0.7071],[0,0.7071,0.7071]]).dot(arrow_dir)
    #h2 = np.array([[1,0,0],[0,0.7071,0.7071],[0,-0.7071,0.7071]]).dot(arrow_dir)
    #head_points = [end_point - 0.001*h2, end_point, end_point - 0.001*h1]
    #vis3d.plot3d(shaft_points, color=teal, tube_radius=.0002)
    #vis3d.plot3d(head_points, color=teal, tube_radius=.0002)
    #
    #vis3d.points(Point(start_point), color=[0,1,0], scale=.001)
    #vis_axes(origin, y_dir)
    vis3d.show(starting_camera_pose=CAMERA_POSE)
    sys.exit()

def noise_vis():
    print 'best actions', np.max(action.metadata['vertex_probs'], axis=0)
    render_3d_scene(env)
    j = 113
    j = np.argmax(action.metadata['vertex_probs'][:,1])
    a = j*policy.toppling_model.n_trials
    b = (j+1)*policy.toppling_model.n_trials
    for i in range(a, b):
        start = policy.toppling_model.vertices[i]
        end = start - .01*policy.toppling_model.push_directions[i]
        vis3d.plot([start, end], color=[0,1,0], radius=.0002)
    vis3d.show(starting_camera_pose=camera_pose())

def parse_args():
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
    parser.add_argument('--save', action='store_true', help='save to a picture rather than opening a window')
    parser.add_argument('--before', action='store_true', help='Whether to show the object before the action')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print '\n\nSaving to file' if args.save else '\n\nDisplaying in a window'

    config = YamlConfig(args.config_filename)
    policy = SingleTopplePolicy(config, use_sensitivity=True)

    if config['debug']:
        random.seed(SEED)
        np.random.seed(SEED)
    
    env = GraspingEnv(config, config['vis'])
    env.reset()
    #env.state.material_props._color = np.array([0.86274509803] * 3)
    #env.state.material_props._color = np.array([.345, .094, .27])
    env.state.material_props._color = np.array([0.5] * 3)
    obj_name = env.state.obj.key
    policy.set_environment(env.environment)

    if False:
        while True:
            env.reset()
            env.render_3d_scene()        
            vis3d.show(starting_camera_pose=CAMERA_POSE)
    
    if args.before:
        # env.state.obj.T_obj_world.translation += np.array([-.01,-.05,.001])
        
        # action = policy.grasping_policy.action(env.state)
        # print 'q', action.q_value
        # gripper(env.gripper(action), action.grasp(env.gripper(action)))
        # vis3d.mesh(env._grippers[GripperType.SUCTION].mesh)
        # color = [0,0,0]
        # mesh = env.state.obj.mesh.copy()
        # mesh.apply_transform(env.state.T_obj_world.matrix)
        # #vis3d.points(Point(np.mean(mesh.vertices, axis=0)), scale=.001)
        # #vis3d.points(Point(mesh.center_mass+np.array([0,.04,.04])), scale=.001)
        # vis3d.mesh(mesh, color=env.state.color)
        # #vis3d.points(env.state.T_obj_world.translation, radius=.001)
        # vis3d.show(starting_camera_pose=CAMERA_POSE)

        env.state.obj.T_obj_world.translation += np.array([-.01,-.05,.001])
        from dexnet.envs import NoActionFoundException
        try:
            action = policy.grasping_policy.action(env.state)
            print 'q', action.q_value
            vis3d.gripper(env.gripper(action), action.grasp(env.gripper(action)))
        except NoActionFoundException as e:
            from dexnet.grasping import GripperType
            vis3d.mesh(env._grippers[GripperType.SUCTION].mesh)

        env.render_3d_scene()        
        vis3d.show(starting_camera_pose=CAMERA_POSE)

    #figure_1()
    #figure_2()
    #figure_3()
    # figure_0()
    action = policy.action(env.state, env)
    #noise_vis()

    if False:
        j = np.argmax(action.metadata['vertex_probs'][:,2])
        R = policy.toppling_model.tipping_point_rotations()[1]
        mesh.apply_transform(R.matrix)
        vis3d.mesh(mesh, color=env.state.color)
        new_point = (R * Point(action.metadata['vertices'][j], 'world')).data
        new_normal = (R * Point(action.metadata['normals'][j], 'world')).data
        vis3d.points(new_point, radius=.001, color=[0,1,0])
        vis3d.plot([new_point, new_point+.01*new_normal], radius=.0001)
        vis3d.show(starting_camera_pose=CAMERA_POSE)
    
    # state.T_obj_world.translation[:2] = np.array([0,0])

    # Visualize
    if args.topple_probs:
        vis3d.figure()
        env.render_3d_scene()
        for vertex, prob in zip(action.metadata['vertices'], action.metadata['topple_probs']):
           color = [min(1, 2*(1-prob)), min(2*prob, 1), 0]
           vis3d.points(Point(vertex, 'world'), scale=.001, color=color)
        # for bottom_point in policy.toppling_model.bottom_points:
        #    vis3d.points(Point(bottom_point, 'world'), scale=.001, color=[0,0,0])
        display_or_save('{}_topple_probs.gif'.format(obj_name))
        #vis3d.figure(bg_color=[0,0,0,0])
        # vis3d.mesh(env.state.mesh,
        #     env.state.T_obj_world.matrix,
        #     color=env.state.color)

        # for vertex, prob in zip(action.metadata['vertices'], action.metadata['topple_probs']):
        #     color = [min(1, 2*(1-prob)), min(2*prob, 1), 0]
        #     vis3d.points(vertex, radius=.001, color=color)
        # for bottom_point in action.metadata['bottom_points']:
        #     vis3d.points(bottom_point, radius=.001, color=[0,0,0])
        # display_or_save('{}_topple_probs.gif'.format(obj_name))

    if True:
        vertex_probs = action.metadata['vertex_probs']
        for edge in range(1,vertex_probs.shape[1]):
            vis3d.figure()
            env.render_3d_scene()
            curr_edge_probs = vertex_probs[:,edge]
            max_curr_edge_prob = np.max(curr_edge_probs)
            for vertex, prob in zip(action.metadata['vertices'], curr_edge_probs):
                if prob > 0:
                    color = [min(1, 2*(1-prob)), min(2*prob, 1), 0]
                    scale = .003 if prob == max_curr_edge_prob else .001
                    vis3d.points(Point(vertex, 'world'), scale=scale, color=color)
            vis3d.points(Point(policy.toppling_model.com, 'world'), scale=.001, color=[0,0,1])
            display_or_save('{}_edge_{}_topple_probs.gif'.format(obj_name, edge))
        # vertex_probs = action.metadata['vertex_probs']
        # for edge in range(1,vertex_probs.shape[1]):
        #     vis3d.figure()
        #     render_3d_scene(env)
        #     curr_edge_probs = vertex_probs[:,edge]
        #     for vertex, prob in zip(action.metadata['vertices'], curr_edge_probs):
        #         if prob > 0:
        #             color = [min(1, 2*(1-prob)), min(2*prob, 1), 0]
        #             vis3d.points(vertex, radius=.001, color=color)
        #     display_or_save('{}_edge_{}_topple_probs.gif'.format(obj_name, edge))

    if args.quality:
        vis3d.figure()
        env.render_3d_scene()
        num_vertices = len(action.metadata['vertices'])
        for i, vertex, q_increase in zip(np.arange(num_vertices), action.metadata['vertices'], action.metadata['quality_increases']):
            topples = action.metadata['final_pose_ind'][i] != 0
            color = [min(1, 2*(1-q_increase)), min(2*q_increase, 1), 0] if topples else [0,0,0]
            vis3d.points(Point(vertex, 'world'), scale=.001, color=color)
        display_or_save('{}_quality_increases.gif'.format(obj_name))
    
    n = len(action.metadata['final_poses'])
    HSV_tuples = [(x*1.0/n, 1, 0.5) for x in range(n)]
    colors = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    colors = np.vstack([np.array([0,0,0]), colors])
    if args.topple_graph:
        vis3d.figure()
        env.render_3d_scene()
        for vertex, pose_ind in zip(action.metadata['vertices'], action.metadata['final_pose_ind']):
            color = colors[pose_ind]
            vis3d.points(Point(vertex, 'world'), scale=.001, color=color)
        vis3d.show(starting_camera_pose=CAMERA_POSE)

        pose_num = 0
        colors = colors[1:]
        for (
            pose, 
            edge_point1, 
            edge_point2, 
            pose_ind
        ) in zip(
            action.metadata['final_poses'], 
            action.metadata['bottom_points'], 
            np.roll(action.metadata['bottom_points'],-1,axis=0), 
            np.arange(n)
        ):
            print 'Pose:', pose_ind
            #env.state.material_props._color = colors[pose_ind]
            env.state.obj.T_obj_world = pose
            
            vis3d.figure()
            env.render_3d_scene()
            #vis3d.points(Point(edge_point1, 'world'), scale=.001, color=np.array([0,0,1]))
            #vis3d.points(Point(edge_point2, 'world'), scale=.001, color=np.array([0,0,1]))
            vis3d.show(starting_camera_pose=CAMERA_POSE) 
