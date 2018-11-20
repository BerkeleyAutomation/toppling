# -*- coding: utf-8 -*-
"""
Copyright ©2017. The Regents of the University of California (Regents). All Rights Reserved.
Permission to use, copy, modify, and distribute this software and its documentation for educational,
research, and not-for-profit purposes, without fee and without a signed licensing agreement, is
hereby granted, provided that the above copyright notice, this paragraph and the following two
paragraphs appear in all copies, modifications, and distributions. Contact The Office of Technology
Licensing, UC Berkeley, 2150 Shattuck Avenue, Suite 510, Berkeley, CA 94720-1620, (510) 643-
7201, otl@berkeley.edu, http://ipira.berkeley.edu/industry-info for commercial licensing opportunities.

IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF
THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF REGENTS HAS BEEN
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF ANY, PROVIDED
HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
"""
"""
Benchmarks a bin picking policy on specific heaps and timesteps from a previous instantiation.
Author: Jeff Mahler
"""
import argparse
import gc
import IPython
import json
import logging
import numpy as np
import os
import pybullet
import random
import ray
import sys
import time
import traceback
from scipy.misc import comb
import math

from autolab_core import Point, RigidTransform, YamlConfig, TensorDataset
import autolab_core.utils as utils
from perception import RenderMode

from dexnet.constants import *
from dexnet.envs import GraspingEnv
from dexnet.envs import GraspAction, ParallelJawGraspAction, SuctionGraspAction, LinearPushAction, NoActionFoundException
from dexnet.visualization import DexNetVisualizer2D as vis2d
from dexnet.visualization import DexNetVisualizer3D as vis3d

from toppling.policies import TopplingPolicy

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
CAMERA_POSE = RigidTransform(CAMERA_ROT, CAMERA_TRANS, from_frame='camera', to_frame='world')

def benchmark_bin_picking_policy(policy,
                                 # input_dataset_path,
                                 # heap_ids,
                                 # timesteps,
                                 # output_dataset_path,
                                 config,
                                 # excluded_heaps_file):
                                 ):
    """ Benchmark a bin picking policy.

    Parameters
    ----------
    policy : :obj:`Policy`
        policy to roll out
    input_dataset_path : str
        path to the input dataset
    heap_ids : list
        integer identifiers for the heaps to re-run
    timesteps : list
        integer timesteps to seed the simulation from
    output_dataset_path : str
        path to store the results
    config : dict
        dictionary-like objects containing parameters of the simulator and visualization
    """
    # read subconfigs
    vis_config = config['vis']
    dataset_config = config['dataset']

    # read parameters
    fully_observed = config['fully_observed']
    steps_per_test_case = config['steps_per_test_case']
    rollouts_per_garbage_collect = config['rollouts_per_garbage_collect']
    debug = config['debug']
    im_height = config['state_space']['camera']['im_height']
    im_width = config['state_space']['camera']['im_width']
    max_obj_per_pile = config['state_space']['object']['max_obj_per_pile']

    if debug:
        random.seed(SEED)
        np.random.seed(SEED)

    # read ids
    # if len(heap_ids) != len(timesteps):
    #     raise ValueError('Must provide same number of heap ids and timesteps')
    # num_rollouts = len(heap_ids)
    num_rollouts = 1
        
    # set dataset params
    tensor_config = dataset_config['tensors']
    fields_config = tensor_config['fields']
    # fields_config['color_ims']['height'] = im_height
    # fields_config['color_ims']['width'] = im_width
    # fields_config['depth_ims']['height'] = im_height
    # fields_config['depth_ims']['width'] = im_width
    fields_config['obj_poses']['height'] = POSE_DIM * max_obj_per_pile
    fields_config['obj_coms']['height'] = POINT_DIM * max_obj_per_pile
    fields_config['obj_ids']['height'] = max_obj_per_pile
    fields_config['bin_distances']['height'] = max_obj_per_pile
    # matrix has (n choose 2) elements in it
    max_distance_matrix_length = int(comb(max_obj_per_pile, 2))
    fields_config['distance_matrix']['height'] = max_distance_matrix_length

    # sample a process id
    proc_id = utils.gen_experiment_id()
    # if not os.path.exists(output_dataset_path):
    #     try:
    #         os.mkdir(output_dataset_path)
    #     except:
    #         logging.warning('Failed to create %s. The dataset path may have been created simultaneously by another process' %(dataset_path))
    # proc_id = 'clustering_2'
    # output_dataset_path = os.path.join(output_dataset_path, 'dataset_%s' %(proc_id))

    # open input dataset
    # logging.info('Opening input dataset: %s' % input_dataset_path)
    # input_dataset = TensorDataset.open(input_dataset_path)
    
    # open output_dataset
    # logging.info('Opening output dataset: %s' % output_dataset_path)
    # dataset = TensorDataset(output_dataset_path, tensor_config)
    # datapoint = dataset.datapoint_template

    # setup logging
    # experiment_log_filename = os.path.join(output_dataset_path, 'dataset_generation.log')
    # formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    # hdlr = logging.FileHandler(experiment_log_filename)
    # hdlr.setFormatter(formatter)
    # logging.getLogger().addHandler(hdlr)
    # config.save(os.path.join(output_dataset_path, 'dataset_generation_params.yaml'))
    
    # key mappings
    # we add the empty string as a mapping because if you don't evaluate dexnet on the 'before' state of the push
    obj_id = 1
    obj_ids = {'': 0}
    action_ids = {
        'ParallelJawGraspAction': 0,
        'SuctionGraspAction': 1,
        'LinearPushAction': 2
    }
    
    # add action ids
    reverse_action_ids = utils.reverse_dictionary(action_ids)
    # dataset.add_metadata('action_ids', reverse_action_ids)
    
    # perform rollouts
    n = 0
    rollout_start = time.time()
    current_heap_id = None
    while n < num_rollouts:
        # create env
        create_start = time.time()
        bin_picking_env = GraspingEnv(config, vis_config)
        create_stop = time.time()
        logging.info('Creating env took %.3f sec' %(create_stop-create_start))

        # perform rollouts
        rollouts_remaining = num_rollouts - n
        for i in range(min(rollouts_per_garbage_collect, rollouts_remaining)):
            # log current rollout
            logging.info('\n')
            if n % vis_config['log_rate'] == 0:
                logging.info('Rollout: %03d' %(n))

            try:    
                # mark rollout status
                data_saved = False
                num_steps = 0
                
                # read heap id
                # heap_id = heap_ids[n]
                # timestep = timesteps[n]
                # while heap_id == current_heap_id:# or heap_id < 81:#[226, 287, 325, 453, 469, 577, 601, 894, 921]: 26
                #     n += 1
                #     heap_id = heap_ids[n]
                #     timestep = timesteps[n]
                push_logger = logging.getLogger('push')
                # push_logger.info('~')
                # push_logger.info('Heap ID %d' % heap_id)
                # current_heap_id = heap_id
                
                # reset env
                reset_start = time.time()
                # bin_picking_env.reset_from_dataset(input_dataset,
                #                                    heap_id,
                #                                    timestep)
                bin_picking_env.reset()
                state = bin_picking_env.state
                environment = bin_picking_env.environment
                if fully_observed:
                    observation = None
                else:
                    observation = bin_picking_env.observation
                policy.set_environment(environment) 
                reset_stop = time.time()

                # add objects to mapping
                for obj_key in state.obj_keys:
                    if obj_key not in obj_ids.keys():
                        obj_ids[obj_key] = obj_id
                        obj_id += 1
                    push_logger.info(obj_key)
                # save id mappings
                reverse_obj_ids = utils.reverse_dictionary(obj_ids)
                # dataset.add_metadata('obj_ids', reverse_obj_ids)
                        
                # store datapoint env params
                # datapoint['heap_ids'] = current_heap_id
                # datapoint['camera_poses'] = environment.camera.T_camera_world.vec
                # datapoint['camera_intrs'] = environment.camera.intrinsics.vec
                # datapoint['robot_poses'] = environment.robot.T_robot_world.vec
            
                # render
                if vis_config['initial_state']:
                    vis3d.figure()
                    bin_picking_env.render_3d_scene()
                    vis3d.pose(environment.robot.T_robot_world)
                    vis3d.show(animate=True)
            
                # observe
                if vis_config['initial_obs']:
                    vis2d.figure()
                    vis2d.imshow(observation, auto_subplot=True)
                    vis2d.show()

                # rollout on current satte
                done = False
                failed = False
                # if isinstance(policy, SingulationFullRolloutPolicy):
                #     policy.reset_num_failed_grasps()
                while not done:
                    if vis_config['step_stats']:
                        logging.info('Heap ID: %s' % heap_id)
                        logging.info('Timestep: %s' % bin_picking_env.timestep)

                    # get action
                    policy_start = time.time()
                    if fully_observed:
                        action = policy.action(state)
                    else:
                        action = policy.action(observation)
                    policy_stop = time.time()
                    logging.info('Composite Policy took %.3f sec' %(policy_stop-policy_start))

                    # render scene before
                    if vis_config['action']:
                        #gripper = bin_picking_env.gripper(action)
                        vis3d.figure()
			            # GRASPINGENV
                        # bin_picking_env.render_3d_scene(render_camera=False, workspace_objs_wireframe=False)
                        bin_picking_env.render_3d_scene()
                        if isinstance(action, GraspAction):
                            vis3d.gripper(gripper, action.grasp(gripper))
                        #if isinstance(action, LinearPushAction):
                        else:
                            # T_start_world = action.T_begin_world * gripper.T_mesh_grasp
                            # T_end_world = action.T_end_world * gripper.T_mesh_grasp
                            #start_point = action.T_begin_world.translation
                            start_point = action['start']
                            #end_point = action.T_end_world.translation
                            end_point = action['end']
                            vec = (end_point - start_point) / np.linalg.norm(end_point-start_point) if np.linalg.norm(end_point-start_point) > 0 else end_point-start_point 
                            #h1 = np.array([[0.7071,-0.7071,0],[0.7071,0.7071,0],[0,0,1]]).dot(vec)
                            #h2 = np.array([[0.7071,0.7071,0],[-0.7071,0.7071,0],[0,0,1]]).dot(vec)
                            arrow_len = np.linalg.norm(start_point - end_point)
                            h1 = (end_point - start_point + np.array([0,0,arrow_len])) / (arrow_len*math.sqrt(2))
                            h2 = (end_point - start_point - np.array([0,0,arrow_len])) / (arrow_len*math.sqrt(2))
                            shaft_points = [start_point, end_point]
                            head_points = [end_point - 0.03*h2, end_point, end_point - 0.03*h1]
                            #vis3d.plot3d(shaft_points, color=[0,0,1])
                            #vis3d.plot3d(head_points, color=[0,0,1])
                            
                            # Displaying all potential topple points
                            for vertex, prob in zip(action['vertices'], action['probabilities']):
                                color = np.array([min(1, 2*(1-prob)), min(2*prob, 1), 0])
                                # color = np.array([0,0,1])
                                vis3d.points(Point(vertex, 'world'), scale=.0005, color=color)

                            for vertex in action['edge_points']:
                                color = np.array([0,0,1])
                                vis3d.points(Point(vertex, 'world'), scale=.0005, color=color)
                            #set_of_lines = action['set_of_lines']
                            #for i, line in enumerate(set_of_lines):
                            #    color = str(bin(i+1))[2:].zfill(3)
                            #    color = np.array([color[2], color[1], color[0]])
                            #    vis3d.plot3d(line, color=color)
                        vis3d.show(starting_camera_pose=CAMERA_POSE)
                        # vis3d.save('/home/mjd3/Pictures/weird_pics/%d_%d_before.png' % (heap_id, bin_picking_env.timestep), starting_camera_pose=CAMERA_POSE)
                    # store datapoint pre-step data
                    j = 0
                    obj_poses = np.zeros(fields_config['obj_poses']['height'])
                    obj_coms = np.zeros(fields_config['obj_coms']['height'])
                    obj_ids_vec = np.iinfo(np.uint32).max * np.ones(fields_config['obj_ids']['height'])
                    for obj_state in state.obj_states:
                        obj_poses[j*POSE_DIM:(j+1)*POSE_DIM] = obj_state.T_obj_world.vec
                        obj_coms[j*POINT_DIM:(j+1)*POINT_DIM] = obj_state.center_of_mass
                        obj_ids_vec[j] = obj_ids[obj_state.key]
                        j += 1
                    action_poses = np.zeros(fields_config['action_poses']['height'])
                    #if isinstance(action, GraspAction):
                    #    action_poses[:7] = action.T_grasp_world.vec
                    #else:
                    #    action_poses[:7] = action.T_begin_world.vec
                    #    action_poses[7:] = action.T_end_world.vec

                    # if isinstance(policy, SingulationMetricsCompositePolicy):
                    #     actual_distance_matrix_length = int(comb(len(state.objs), 2))
                    #     bin_distances = np.append(action.metadata['bin_distances'], 
                    #                               np.zeros(max_obj_per_pile-len(state.objs))
                    #                             )
                    #     distance_matrix = np.append(action.metadata['distance_matrix'], 
                    #                                 np.zeros(max_distance_matrix_length - actual_distance_matrix_length)
                    #                             )
                    #     datapoint['bin_distances'] = bin_distances
                    #     datapoint['distance_matrix'] = distance_matrix
                    #     datapoint['T_begin_world'] = action.T_begin_world.matrix
                    #     datapoint['T_end_world'] = action.T_end_world.matrix
                    #     datapoint['parallel_jaw_best_q_value'] = action.metadata['parallel_jaw_best_q_value']
                    #     # datapoint['parallel_jaw_mean_q_value'] = action.metadata['parallel_jaw_mean_q_value']
                    #     # datapoint['parallel_jaw_num_grasps'] = action.metadata['parallel_jaw_num_grasps']
                    #     datapoint['suction_best_q_value'] = action.metadata['suction_best_q_value']
                    #     # datapoint['suction_mean_q_value'] = action.metadata['suction_mean_q_value']
                    #     # datapoint['suction_num_grasps'] = action.metadata['suction_num_grasps']
                    #     # logging.info('Suction Q: %f, PJ Q: %f' % (action.metadata['suction_q_value'], action.metadata['parallel_jaw_q_value']))
                    #     # datapoint['obj_index'] = action.metadata['obj_index']

                    #     # datapoint['parallel_jaw_best_q_value_single'] = action.metadata['parallel_jaw_best_q_value_single']
                    #     # datapoint['suction_best_q_value_single'] = action.metadata['suction_best_q_value_single']
                    #     datapoint['singulated_obj_index'] = action.metadata['singulated_obj_index']
                    #     datapoint['parallel_jaw_grasped_obj_index'] = obj_ids[action.metadata['parallel_jaw_grasped_obj_key']]
                    #     datapoint['suction_grasped_obj_index'] = obj_ids[action.metadata['suction_grasped_obj_key']]
                    # else:
                    #     datapoint['bin_distances'] = np.zeros(max_obj_per_pile)
                    #     datapoint['distance_matrix'] = np.zeros(max_distance_matrix_length)
                    #     datapoint['T_begin_world'] = np.zeros((4,4))
                    #     datapoint['T_end_world'] = np.zeros((4,4))
                    #     datapoint['parallel_jaw_best_q_value'] = -1
                    #     datapoint['suction_best_q_value'] = -1
                    #     datapoint['singulated_obj_index'] = -1
                    #     datapoint['parallel_jaw_grasped_obj_index'] = -1
                    #     datapoint['suction_grasped_obj_index'] = -1

                    # policy_id = 0
                    # if 'policy_id' in action.metadata.keys():
                    #     policy_id = action.metadata['policy_id']
                    # greedy_q_value = 0
                    # if 'greedy_q_value' in action.metadata.keys():
                    #     greedy_q_value = action.metadata['greedy_q_value']
                        
                    # datapoint['timesteps'] = bin_picking_env.timestep
                    # datapoint['obj_poses'] = obj_poses
                    # datapoint['obj_coms'] = obj_coms
                    # datapoint['obj_ids'] = obj_ids_vec
                    # # if bin_picking_env.render_mode == RenderMode.RGBD:
                    # #     color_data = observation.color.raw_data
                    # #     depth_data = observation.depth.raw_data
                    # # elif bin_picking_env.render_mode == RenderMode.DEPTH:
                    # #     color_data = np.zeros(observation.shape).astype(np.uint8)
                    # #     depth_data = observation.raw_data
                    # # elif bin_picking_env.render_mode == RenderMode.COLOR:
                    # #     color_data = observation.raw_data
                    # #     depth_data = np.zeros(observation.shape)
                    # # datapoint['color_ims'] = color_data
                    # # datapoint['depth_ims'] = depth_data
                    # datapoint['action_ids'] = action_ids[type(action).__name__]
                    # datapoint['action_poses'] = action_poses
                    # datapoint['policy_ids'] = policy_id
                    # datapoint['greedy_q_values'] = greedy_q_value
                    # datapoint['pred_q_values'] = action.q_value
                    
                    # step the policy
                    #observation, reward, done, info = bin_picking_env.step(action)
                    #state = bin_picking_env.state
                    state.objs[0].T_obj_world = action['final_state']

                    # if isinstance(policy, SingulationFullRolloutPolicy):
                    #     policy.grasp_succeeds(info['grasp_succeeds'])
        
                    # debugging info
                    if vis_config['step_stats']:
                        logging.info('Action type: %s' %(type(action).__name__))
                        logging.info('Action Q-value: %.3f' %(action.q_value))
                        logging.info('Reward: %d' %(reward))
                        logging.info('Policy took %.3f sec' %(policy_stop-policy_start))
                        logging.info('Num objects remaining: %d' %(bin_picking_env.num_objects))
                        if info['cleared_pile']:
                            logging.info('Cleared pile!')
                        
                    # # store datapoint post-step data
                    # datapoint['rewards'] = reward
                    # datapoint['grasp_metrics'] = info['grasp_metric']
                    # datapoint['collisions'] = 1 * info['collides']
                    # datapoint['collisions_with_env'] = 1 * info['collides_with_static_obstacles']
                    # datapoint['grasped_obj_ids'] = obj_ids[info['grasped_obj_key']]
                    # datapoint['cleared_pile'] = 1 * info['cleared_pile']

                    # # store datapoint
                    # # dataset.add(datapoint)
                    # data_saved = True    
                    
                    # render observation
                    if vis_config['obs']:
                        vis2d.figure()
                        vis2d.imshow(observation, auto_subplot=True)
                        vis2d.show()
        
                    # render scene after
                    if vis_config['state']:
                        vis3d.figure()
                        bin_picking_env.render_3d_scene(render_camera=False)
                        vis3d.show(starting_camera_pose=CAMERA_POSE)
                        # vis3d.save('/home/mjd3/Pictures/weird_pics/%d_%d_after.png' % (heap_id, bin_picking_env.timestep), starting_camera_pose=CAMERA_POSE)
                    state.objs[0].T_obj_world = action['tmpR']
                    vis3d.figure()
                    bin_picking_env.render_3d_scene(render_camera=False)
                    vis3d.show(starting_camera_pose=CAMERA_POSE)
                    state.objs[0].T_obj_world = action['final_state']
                    # increment the number of steps
                    num_steps += 1
                    if num_steps >= steps_per_test_case:
                        done = True
                        
            except NoActionFoundException as e:
                logging.warning('The policy failed to plan an action!')
                done = True                    
            except Exception as e:
                # log an error
                logging.warning('Rollout failed!')
                logging.warning('%s' %(str(e)))
                logging.warning(traceback.print_exc())
                # if debug:
                #     raise
                
                # reset env
                del bin_picking_env
                gc.collect()
                bin_picking_env = BinPickingEnv(config, vis_config)

                # terminate current rollout
                failed = True
                done = True

            # update test case id
            n += 1
            # dataset.flush()
            # logging.info("\n\nflushing")
            # logging.info("exiting")
            # sys.exit()
                
        # garbage collect
        del bin_picking_env
        gc.collect()

    # return the dataset 
    # dataset.flush()

    # log time
    rollout_stop = time.time()
    logging.info('Rollouts took %.3f sec' %(rollout_stop-rollout_start))

    return dataset

def run_sequential_bin_picking_benchmark(
                                         # input_dataset_path,
                                         # heap_ids,
                                         # timesteps,
                                         # output_dataset_path,
                                         config_filename,
                                         # excluded_heaps_file,
                                         num_rollouts=None):
    # open config file
    logging.getLogger().setLevel(logging.INFO)
    config = YamlConfig(config_filename)
    if num_rollouts is not None:
        config['num_rollouts'] = num_rollouts

    # load policy
    policy = TopplingPolicy()

    # perform rollouts
    logging.info('Rolling out policy')
    dataset = benchmark_bin_picking_policy(policy,
                                           # input_dataset_path,
                                           # heap_ids,
                                           # timesteps,
                                           # output_dataset_path,
                                           config,
                                           # excluded_heaps_file)
                                           )
    return dataset

@ray.remote
def benchmark_bin_picking_policy_in_parallel(input_dataset_path,
                                             heap_ids,
                                             timesteps,
                                             output_dataset_path,
                                             config_filename,
                                             num_rollouts):
    """ Rollout a bin picking policy in parallel using ray.

    Parameters
    ----------
    policy : :obj:`Policy`
        policy to roll out
    dataset_path : str
        path to store the dataset
    config_filename : str
        filename of a Yaml configuration file
    num_rollouts : int
        number of rollouts to execute

    Returns
    -------
    str
       filename of dataset
    """
    dataset = run_sequential_bin_picking_benchmark(input_dataset_path,
                                                   heap_ids,
                                                   timesteps,
                                                   output_dataset_path,
                                                   config_filename,
                                                   num_rollouts=num_rollouts)
    return dataset.filename

def run_parallel_bin_picking_benchmark(input_dataset_path,
                                       heap_ids,
                                       timesteps,
                                       output_dataset_path,
                                       config_filename):
    raise NotImplementedError('Cannot run in parallel. Need to split up the heap ids and timesteps')

    # load config
    config = YamlConfig(config_filename)

    # init ray
    ray_config = config['ray']
    num_cpus = ray_config['num_cpus']
    ray.init(num_cpus=num_cpus,
             redirect_output=ray_config['redirect_output'])
    
    # rollouts
    num_rollouts = config['num_rollouts'] // num_cpus
    dataset_ids = [rollout_bin_picking_policy_in_parallel.remote(dataset_path, config_filename, num_rollouts) for i in range(num_cpus)]
    dataset_filenames = ray.get(dataset_ids)
    if len(dataset_filenames) == 0:
        return
    
    # merge datasets    
    subproc_dataset = TensorDataset.open(dataset_filenames[0])
    tensor_config = subproc_dataset.config


    # open dataset
    dataset = TensorDataset(dataset_path, tensor_config)
    dataset.add_metadata('action_ids', subproc_dataset.metadata['action_ids'])

    # add datapoints
    obj_id = 0
    heap_id = 0
    obj_ids = {}
    for dataset_filename in dataset_filenames:
        logging.info('Aggregating data from %s' %(dataset_filename))
        j = 0
        subproc_dataset = TensorDataset.open(dataset_filename)
        subproc_obj_ids = subproc_dataset.metadata['obj_ids']
        for datapoint in subproc_dataset:
            if j > 0 and datapoint['timesteps'] == 0:
                heap_id += 1
                
            # modify object ids
            for i in range(datapoint['obj_ids'].shape[0]):
                subproc_obj_id = datapoint['obj_ids'][i]
                if subproc_obj_id != np.uint32(-1):
                    subproc_obj_key = subproc_obj_ids[str(subproc_obj_id)]
                    if subproc_obj_key not in obj_ids.keys():
                        obj_ids[subproc_obj_key] = obj_id
                        obj_id += 1
                    datapoint['obj_ids'][i] = obj_ids[subproc_obj_key]

            # modify grasped obj id
            subproc_grasped_obj_id = datapoint['grasped_obj_ids']
            grasped_obj_key = subproc_obj_ids[str(subproc_grasped_obj_id)]
            datapoint['grasped_obj_ids'] = obj_ids[grasped_obj_key]

            # modify heap id
            datapoint['heap_ids'] = heap_id
                
            # add datapoint to dataset
            dataset.add(datapoint)
            j += 1
            
    # write to disk        
    obj_ids = utils.reverse_dictionary(obj_ids)
    dataset.add_metadata('obj_ids', obj_ids)
    dataset.flush()
    
if __name__ == '__main__':
    # initialize logging
    logging.getLogger().setLevel(logging.INFO)

    # parse args
    parser = argparse.ArgumentParser(description='Rollout a policy for bin picking in order to evaluate performance')
    # parser.add_argument('input_dataset_path', type=str, default=None, help='directory containing the dataset to use for seeding heaps')
    # parser.add_argument('--input_test_case_file', '-itf', type=str, default=None, help='json file containing the heap ids and timesteps of test cases')
    # parser.add_argument('--input_test_case_key', '-itk', type=str, default=None, help='key within the json file for the test cases')
    # parser.add_argument('output_dataset_path', type=str, default=None, help='directory to store a dataset containing the results of rollouts')
    parser.add_argument('--config_filename', type=str, default=None, help='configuration file to use')
    # parser.add_argument('--excluded_heaps_file', type=str, default=None, help='file with heap numbers to exclude')
    args = parser.parse_args()
    # input_dataset_path = args.input_dataset_path
    # input_test_case_file = args.input_test_case_file
    # input_test_case_key = args.input_test_case_key
    # # output_dataset_path = args.output_dataset_path
    config_filename = args.config_filename
    # # excluded_heaps_file = args.excluded_heaps_file
    
    # handle config filename
    if config_filename is None:
        config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/benchmark_topple_policy_graspingenv.yaml')
    # if excluded_heaps_file is None:
    #     excluded_heaps_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #                                        '..',
    #                                        'data/dn4_graspability/greedy_bad_heaps_v5.txt')

    # # turn relative paths absolute
    # if not os.path.isabs(config_filename):
    #     config_filename = os.path.join(os.getcwd(), config_filename)

    # # load test cases
    # if input_test_case_file is None:
    #     test_cases = json.load(open(input_dataset_path+'metadata.json', 'r'))
    #     heap_ids = test_cases['failure_cases']['heap_ids']
    #     timesteps = test_cases['failure_cases']['timesteps']
    # else:
    #     test_cases = json.load(open(input_test_case_file, 'r'))
    #     heap_ids = test_cases[input_test_case_key]['heap_ids']
    #     timesteps = test_cases[input_test_case_key]['timesteps']

    # # perform rollouts
    # run_sequential_bin_picking_benchmark(input_dataset_path,
    #                                      heap_ids,
    #                                      timesteps,
    #                                      output_dataset_path,
    #                                      config_filename,
    #                                      excluded_heaps_file)
    run_sequential_bin_picking_benchmark(config_filename)
