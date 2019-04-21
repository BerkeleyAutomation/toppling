import argparse
import numpy as np
import os
import json
import random
import logging
import matplotlib.pyplot as plt
from sklearn import metrics
from copy import copy
from random import shuffle

from autolab_core import Point, RigidTransform, YamlConfig, TensorDataset
from dexnet.constants import *
from dexnet.envs import GraspingEnv
from dexnet.visualization import DexNetVisualizer3D as vis3d
from toppling.models import TopplingModel
from toppling import camera_pose, is_equivalent_pose, pose_diff, pose_angle

CAMERA_POSE = camera_pose()
SEED = 107
<<<<<<< HEAD
NUM_MODELS = 100
=======
NUM_MODELS = 500
>>>>>>> e9cb2563ba5681b50511c256f12cfcf742cf406a
NUM_MODELS_TO_KEEP = 10
NUM_PER_DATAPOINT = 10

def to_rigid(mat):
    rot, trans = RigidTransform.rotation_and_translation_from_matrix(mat)
    return RigidTransform(rot, trans, 'obj', 'world')

def precision_recall(pos_probs, neg_probs, num_toppled, thresh):
    true_pos = np.sum(pos_probs >= thresh)
    false_pos = np.sum(neg_probs >= thresh)
    precision = true_pos / float(true_pos + false_pos)
    recall = true_pos / float(num_toppled)
    return thresh, precision, recall

def evaluate_single_model(model, datasets, obj_id_to_keys, env, use_sensitivity):
    tvs, l1s, topple_true, topple_pred, pose_true, pose_pred = [], [], [], [], [], []
    total_datapoints = 0
    for dataset, obj_id_to_key in zip(datasets, obj_id_to_keys):
        total_datapoints += dataset.num_datapoints
        per_obj_tvs, per_obj_l1s = [], []
        per_obj_topple_true, per_obj_topple_pred, per_obj_pose_true, per_obj_pose_pred = [], [], [], []
        for i in range(dataset.num_datapoints):
            datapoint = dataset.datapoint(i)

            dataset_name, key = obj_id_to_key[str(datapoint['obj_id'])].split(KEY_SEP_TOKEN)
            obj = env._state_space._database.dataset(dataset_name)[key]
            orig_pose = to_rigid(datapoint['obj_pose'])
            obj.T_obj_world = orig_pose
            env.state.obj = obj
            
            model.load_object(env.state)
            predicted_poses, vertex_probs, _ = model.predict(
                [datapoint['vertex']], 
                [datapoint['normal']], 
                [-datapoint['normal']], # push dir
                use_sensitivity=use_sensitivity
            )
            vertex_probs = vertex_probs[0]
            per_obj_topple_pred.extend([1-vertex_probs[0]] * NUM_PER_DATAPOINT)

            empirical_dist = []
            for i in range(NUM_PER_DATAPOINT):
                actual_pose_mat = datapoint['actual_poses'][i*4:(i+1)*4]
                rot, trans = RigidTransform.rotation_and_translation_from_matrix(actual_pose_mat)
                pose_to_add = RigidTransform(rot, trans, 'obj', 'world')

                per_obj_topple_true.append(0 if is_equivalent_pose(pose_to_add, orig_pose) else 1)

                found = False
                for i in range(len(empirical_dist)):
                    actual_pose, actual_prob = empirical_dist[i]
                    if is_equivalent_pose(actual_pose, pose_to_add):
                        empirical_dist[i][1] += 1 / float(NUM_PER_DATAPOINT)
                        found = True
                        break
                if not found:
                    empirical_dist.append([pose_to_add, 1 / float(NUM_PER_DATAPOINT)])

                found = False
                for predicted_pose, predicted_prob in zip(predicted_poses, vertex_probs):
                    if is_equivalent_pose(pose_to_add, predicted_pose):
                        found = True
                        per_obj_pose_true.append(1)
                    else:
                        per_obj_pose_true.append(0)
                    per_obj_pose_pred.append(predicted_prob)
                if not found:
                    per_obj_pose_true.append(1)
                    per_obj_pose_pred.append(0)
            
            total_variation, l1 = 1, []
            for empirical_pose, empirical_prob in empirical_dist:
                for predicted_pose, predicted_prob in zip(predicted_poses, vertex_probs):
                    if is_equivalent_pose(empirical_pose, predicted_pose):
                        total_variation = min(total_variation, abs(empirical_prob - predicted_prob))
                        l1.append(abs(empirical_prob - predicted_prob))
                        break
            l1 = np.mean(l1) if len(l1) > 0 else 0
            per_obj_tvs.append(total_variation)
            per_obj_l1s.append(l1)
            # avg_precisions.append(metrics.average_precision_score(pose_y_true, pose_y_pred))
            # for i in range(10):
            #     actual_pose_mat = datapoint['actual_poses'][i*4:(i+1)*4]
            #     rot, trans = RigidTransform.rotation_and_translation_from_matrix(actual_pose_mat)
            #     actual_pose = RigidTransform(rot, trans, 'obj', 'world')

            #     y_true.append(0 if is_equivalent_pose(orig_pose, actual_pose) else 1)
            #     y_pred.append(1-vertex_probs[0])

            #     #  Cross Entropy
            #     # q_x = 0
            #     # for predicted_pose, prob in zip(predicted_poses, vertex_probs):
            #     #     if is_equivalent_pose(actual_pose, predicted_pose):
            #     #         q_x = prob
            #     #         break
            #     # if q_x == 0:
            #     #     counter += 1
            #     #     # env.render_3d_scene()
            #     #     # vis3d.show(title='before', starting_camera_pose=CAMERA_POSE)
            #     #     # env.state.obj.T_obj_world = actual_pose
            #     #     # env.render_3d_scene()
            #     #     # vis3d.show(title='after', starting_camera_pose=CAMERA_POSE)
            #     #     # for predicted_pose, prob in zip(predicted_poses, vertex_probs):
            #     #     #     env.state.obj.T_obj_world = predicted_pose
            #     #     #     env.render_3d_scene()
            #     #     #     title = '{}, pose diff: {}, angle diff: {}, prob: {}'.format(
            #     #     #         is_equivalent_pose(actual_pose, predicted_pose),
            #     #     #         pose_diff(actual_pose, predicted_pose),
            #     #     #         pose_angle(actual_pose, predicted_pose),
            #     #     #         prob
            #     #     #     )
            #     #     #     vis3d.show(title=title, starting_camera_pose=CAMERA_POSE)
            #     #     # env.state.obj.T_obj_world = to_rigid(datapoint['obj_pose'])
            #     # q_x = max(q_x, 1e-5)
            #     # cross_entropy -= .1 * np.log(q_x) # p(x) * log(q(x))
            #     # n += .1
        if len(per_obj_tvs) < 10:
            per_obj_tvs.extend([0]*(10-len(per_obj_tvs)))
            per_obj_l1s.extend([0]*(10-len(per_obj_l1s)))
        tvs.append(per_obj_tvs)
        l1s.append(per_obj_l1s)

        if len(per_obj_tvs) < 10*NUM_PER_DATAPOINT:
            per_obj_topple_true.extend([0]*(10*NUM_PER_DATAPOINT-len(per_obj_topple_true)))
            per_obj_topple_pred.extend([0]*(10*NUM_PER_DATAPOINT-len(per_obj_topple_pred)))
        topple_true.append(per_obj_topple_true)
        topple_pred.append(per_obj_topple_pred)

        if len(per_obj_pose_true) < 600:
            per_obj_pose_true.extend([0]*(600-len(per_obj_pose_true)))
            per_obj_pose_pred.extend([0]*(600-len(per_obj_pose_pred)))
        pose_true.append(per_obj_pose_true)
        pose_pred.append(per_obj_pose_pred)

    # precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    # aucs.append(metrics.auc(recall, precision))
    # prs.append((precision, recall, model))
    # return np.mean(train_tvs), np.mean(test_tvs), np.mean(combined_tvs), np.mean(train_l1s), np.mean(test_l1s), np.mean(combined_l1s)
    return np.array(tvs), np.array(l1s), np.array(topple_true), np.array(topple_pred), np.array(pose_true), np.array(pose_pred)

# def run_sweep(models, datasets, obj_id_to_keys, test_set, env, use_sensitivity):
#     train_tvs, test_tvs, combined_tvs = [], [], []
#     train_l1s, test_l1s, combined_l1s = [], [], []
#     for model in models:
#         (
#             train_tv, test_tv, combined_tv, 
#             train_l1, test_l1, combined_l1
#         ) = evaluate_single_model(
#             model, datasets, 
#             obj_id_to_keys, test_set, 
#             env, use_sensitivity
#         )
#         logger.info('{} Train TV: {} Test TV: {}, Combined TV: {} Train L1: {}, Test L1: {} Combined L1: {}'.format(
#             model.readable_str(), 
#             train_tv, test_tv, combined_tv, 
#             train_l1, test_l1, combined_l1
#         ))
#         train_tvs.append(train_tv)
#         test_tvs.append(test_tv)
#         train_l1s.append(train_l1)
#         test_l1s.append(test_l1)
#         combined_tvs.append(combined_tv)
#         combined_l1s.append(combined_l1)

#     logger.info('\nBest TV Models:')
#     best_model_idxs = np.argsort(combined_tvs)[:NUM_MODELS_TO_KEEP]
#     for best_model_idx in best_model_idxs:
#         model, train_tv, test_tv = models[best_model_idx], train_tvs[best_model_idx], test_tvs[best_model_idx]
#         combined_tv, combined_l1 = combined_tvs[best_model_idx], combined_l1s[best_model_idx]
#         logger.info('{} Train TV: {} Test TV: {}, Combined TV: {} Train L1: {}, Test L1: {} Combined L1: {}'.format(
#             model.readable_str(), 
#             train_tv, test_tv, combined_tv, 
#             train_l1, test_l1, combined_l1
#         ))

#     logger.info('\nBest L1 Models:')
#     best_model_idxs = np.argsort(combined_l1s)[:NUM_MODELS_TO_KEEP]
#     for best_model_idx in best_model_idxs:
#         model, train_tv, test_tv = models[best_model_idx], train_tvs[best_model_idx], test_tvs[best_model_idx]
#         combined_tv, combined_l1 = combined_tvs[best_model_idx], combined_l1s[best_model_idx]
#         logger.info('{} Train TV: {} Test TV: {}, Combined TV: {} Train L1: {}, Test L1: {} Combined L1: {}'.format(
#             model.readable_str(), 
#             train_tv, test_tv, combined_tv, 
#             train_l1, test_l1, combined_l1
#         ))

def pr_curve(datasets, obj_id_to_keys, env):
    model_config = config['model']
    model_config['ground_friction_coeff'] = 0.439
    model_config['finger_friction_coeff'] = .9259
    model_config['finger_sigma'] = .000415
    model_config['push_direction_sigma'] = .07074
    model = TopplingModel(model_config)
    _, _, avg_precision, y_true, y_pred, pose_y_true, pose_y_pred = evaluate_single_model(model, datasets, obj_id_to_keys, env, True)
    precision, recall, thresh = metrics.precision_recall_curve(y_true, y_pred)

    # plt.style.use('seaborn-darkgrid')
    # plt.rcParams.update({'font.size': 20})
    plt.plot(recall, precision, label=model, linewidth=5.0)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.show()

def plot_pr_curve(models, datasets, obj_id_to_keys, env, use_sensitivity, label):
    y_trues, y_preds, avg_precisions = [], [], []
    for model in models:
        _, _, _, y_true, y_pred, pose_y_true, pose_y_pred = evaluate_single_model(model, datasets, obj_id_to_keys, env, use_sensitivity)
        avg_precisions.append(metrics.average_precision_score(y_true, y_pred))
        y_trues.append(y_true)
        y_preds.append(y_pred)
    best_model = np.argmax(avg_precisions)
    precision, recall, thresh = metrics.precision_recall_curve(y_trues[best_model], y_preds[best_model])
    plt.plot(recall, precision, label=label, linewidth=5.0)

def run_sweep_old(models, datasets, obj_id_to_keys, env, use_sensitivity, plot_pr=False):
    def compute_cv(metrics, idx, use_max):
        metrics = metrics[:,idx]

        folds = np.array_split(metrics, K, axis=1)
        cv = []
        for k in range(K):
            train = []
            for i in range(K):
                if i != k:
                    train.append(folds[i])
            train = np.hstack(train)
            test = folds[k]
            avg_train = np.mean(train, axis=1)
            avg_test = np.mean(test, axis=1)

            if use_max:
                best_model_idx = np.argmax(avg_train)
            else:
                best_model_idx = np.argmin(avg_train)
            cv.append(avg_test[best_model_idx])
        return np.mean(cv)
    def compute_pr_cv(pose_y_trues, pose_y_preds, idx):
        pose_y_trues = pose_y_trues[:,idx]
        pose_y_preds = pose_y_preds[:,idx]

        folds_y_true = np.array_split(pose_y_trues, K, axis=1)
        folds_y_pred = np.array_split(pose_y_preds, K, axis=1)
        cv = []
        for k in range(K):
            train_y_trues, train_y_preds = [], []
            for i in range(K):
                if i != k:
                    train_y_true.append(folds_y_true[i])
                    train_y_pred.append(folds_y_pred[i])
            train_y_trues = np.hstack(train_y_trues)
            train_y_preds = np.hstack(train_y_preds)
            test_y_true = folds_y_true[k]
            test_y_pred = folds_y_pred[k]

            avg_train = []
            for train_y_true, train_y_pred in zip(train_y_trues, train_y_preds):
                avg_train = metrics.average_precision_score(train_y_true, train_y_pred)
            
            best_model_idx = np.argmin(avg_train)
            if np.sum(test_y_true[best_model]) == 0:
                test_precision = 1
            else:
                test_precision = metrics.average_precision_score(test_y_true[best_model_idx], test_y_pred[best_model_idx])

            cv.append(test_precision)
        return np.mean(cv)

    tvs, l1s, avg_ps = [], [], []
    best_avg_precision = 0
    for model in models:
        tv, l1, avg_p, (avg_precision, _, _) = evaluate_single_model(model, datasets, obj_id_to_keys, env, use_sensitivity)
        logger.info('{} TV: {} L1: {} Pose MAP: {} MAP: {}'.format(model.readable_str(), np.mean(tv), np.mean(l1), np.mean(avg_p), avg_precision))
        tvs.append(tv)
        l1s.append(l1)
        avg_ps.append(avg_p)
        # pose_y_trues.append(pose_y_true)
        # pose_y_preds.append(pose_y_pred)
        if avg_precision > best_avg_precision:
            best_avg_precision = avg_precision
        #     best_avg_precision = avg_precision
        #     best_y_true = y_true
        #     best_y_pred = y_pred
    tvs = np.vstack(tvs)
    l1s = np.vstack(l1s)
    avg_ps = np.vstack(avg_ps)

    idx = np.arange(tvs.shape[1])
    shuffle(idx)
    tv_cv = compute_cv(tvs, idx, False)
    l1_cv = compute_cv(l1s, idx, False)
    map_cv = compute_cv(avg_ps, idx, True)
    # map_cv = compute_pr_cv(pose_y_trues, pose_y_preds, idx)
    
    if plot_pr:
        pr_curve(best_y_true, best_y_pred)
    logger.info('Avg CV TV: {}, Avg CV L1: {}, Avg CV (Avg Precision) {}, Best Avg Precision: {}'.format(tv_cv, l1_cv, map_cv, best_avg_precision))

def run_sweep(models, datasets, obj_id_to_keys, env, use_sensitivity, plot_pr=False):
    def compute_cv(metrics):
        cv, idxs = [], []
        num_objs = metrics.shape[1]
        for k in range(num_objs):
            train = []
            for i in range(num_objs):
                if i != k:
                    train.append(metrics[:,i])
            train = np.hstack(train)
            test = metrics[:,k]
            avg_train = np.mean(train, axis=1)
            avg_test = np.mean(test, axis=1)

            best_model_idx = np.argmin(avg_train)
            idxs.append(best_model_idx)
            cv.append(avg_test[best_model_idx])
        return np.mean(cv), idxs

    def compute_map_cv(y_true, y_pred):
        cv, idxs = [], []
        num_objs = y_true.shape[1]
        for k in range(num_objs):
            y_true_train, y_pred_train = [], []
            for i in range(num_objs):
                if i != k:
                    y_true_train.append(y_true[:,i])
                    y_pred_train.append(y_pred[:,i])
            y_true_train = np.hstack(y_true_train)
            y_pred_train = np.hstack(y_pred_train)
            y_true_test = y_true[:,k]
            y_pred_test = y_pred[:,k]

            train_maps = []
            for t, p in zip(y_true_train, y_pred_train):
                train_maps.append(metrics.average_precision_score(t, p))

            best_model_idx = np.argmax(train_maps)
            if np.sum(y_true_test[best_model_idx]) == 0:
                test_precision = 1
            else:
                test_precision = metrics.average_precision_score(y_true_test[best_model_idx], y_pred_test[best_model_idx])

            idxs.append(best_model_idx)
            cv.append(test_precision)
        return np.mean(cv), idxs

    tvs, l1s, topple_trues, topple_preds, pose_trues, pose_preds = [], [], [], [], [], []
    for i, model in enumerate(models):
        tv, l1, topple_true, topple_pred, pose_true, pose_pred = evaluate_single_model(model, datasets, obj_id_to_keys, env, use_sensitivity)
        if np.sum(pose_true) == 0:
            pose_map = 1
        else:
            pose_map = metrics.average_precision_score(pose_true.flatten(), pose_pred.flatten())
        if np.sum(topple_true) == 0:
            topple_map = 1
        else:
            topple_map = metrics.average_precision_score(topple_true.flatten(), topple_pred.flatten())

        logger.info('{}: {} TV: {} L1: {} Pose MAP: {} MAP: {}'.format(
            i, model.readable_str(),
            np.mean(tv),
            np.mean(l1),
            pose_map, topple_map
        ))
        tvs.append(tv)
        l1s.append(l1)
        topple_trues.append(topple_true)
        topple_preds.append(topple_pred)
        pose_trues.append(pose_true)
        pose_preds.append(pose_pred)
    tvs = np.array(tvs)
    l1s = np.array(l1s)
    topple_trues = np.array(topple_trues)
    topple_preds = np.array(topple_preds)
    pose_trues = np.array(pose_trues)
    pose_preds = np.array(pose_preds)
    
    tv_cv, idxs = compute_cv(tvs)
    logger.info('tv best models'+str(idxs))
    l1_cv = compute_cv(l1s)
    logger.info('l1 best models'+str(idxs))
    topple_cv, idxs = compute_map_cv(topple_trues, topple_preds)
    logger.info('topple best models'+str(idxs))
    pose_cv, idxs = compute_map_cv(pose_trues, pose_preds)
    logger.info('poses best modeos'+str(idxs))
    logger.info('TV CV: {}, L1 CV: {}, Topple MAP CV: {}, Pose MAP CV: {}'.format(tv_cv, l1_cv, topple_cv, pose_cv))

def parse_args():
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/benchmark_topple_policy_graspingenv.yaml'
    )
    parser = argparse.ArgumentParser(description='Rollout a policy for bin picking in order to evaluate performance')
    parser.add_argument('--config_filename', type=str, default=default_config_filename, help='configuration file to use')
    parser.add_argument('--datasets', type=str, default='/nfs/diskstation/db/toppling/experiments/', help='Which dataset to load toppling experiments from')
    parser.add_argument('-logfile', type=str, default='/nfs/diskstation/db/toppling/param_sweep_cv.log')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logger = logging.getLogger('toppling')
    hdlr = logging.FileHandler(args.logfile)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    
    config = YamlConfig(args.config_filename)
    if config['debug']:
        random.seed(SEED)
        np.random.seed(SEED)

    models, baseline_models = [], []
    model_config = config['model']
    for _ in range(NUM_MODELS):
        model_config['baseline'] = 0
        model_config['ground_friction_coeff'] = np.random.uniform(.1, .7)
        model_config['finger_friction_coeff'] = np.random.uniform(.3, 1)
        model_config['finger_sigma'] = np.random.uniform(.0003, .0009)
        model_config['push_direction_sigma'] = np.random.uniform(.03, .09)
        models.append(TopplingModel(model_config))
        model_config['baseline'] = 1
        baseline_models.append(TopplingModel(model_config))
    datasets, obj_id_to_keys = [], []
    for dataset_name in os.listdir(args.datasets):
        dataset_name = dataset_name.split(' ')[0]
        dataset_path = os.path.join(args.datasets, dataset_name)
        datasets.append(TensorDataset.open(dataset_path))
        with open(os.path.join(dataset_path, "obj_keys.json"), "r") as read_file:
            obj_id_to_keys.append(json.load(read_file))
    env = GraspingEnv(config, config['vis'])
    env.reset()

    total_datapoints = np.sum([d.num_datapoints for d in datasets])
    print 'total', total_datapoints
    datapoint_ordering = np.arange(total_datapoints)
    shuffle(datapoint_ordering)

    # pr_curve(datasets, obj_id_to_keys, env)
    # sys.exit()

    logger.info('Baseline')
    run_sweep(baseline_models, datasets, obj_id_to_keys, env, False)
    logger.info('\n\n\nWith Rotations')
    run_sweep(models, datasets, obj_id_to_keys, env, False)
    logger.info('\n\n\nWith Robustness')
    run_sweep(baseline_models, datasets, obj_id_to_keys, env, True)
    logger.info('\n\n\nWith Rotations and Robustness')
    run_sweep(models, datasets, obj_id_to_keys, env, True)

    # plt.style.use('seaborn-darkgrid')
    # plot_pr_curve(baseline_models, datasets, obj_id_to_keys, env, False, 'Baseline')
    # plot_pr_curve(models, datasets, obj_id_to_keys, env, False, 'Baseline + Rotations')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.legend(frameon=True, prop={'size': 16})
    # plt.show()
    # plt.savefig('/home/chriscorrea14/param_sweep.png')



    
        

    # plt.style.use('seaborn-darkgrid')
    # plt.rcParams.update({'font.size': 30})
    # plt.figure(figsize=(20,15))
    # for precision, recall, model in prs:
    #     plt.plot(recall, precision, label=model, linewidth=5.0)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.legend(frameon=True, prop={'size': 16})
    # plt.savefig('/home/chriscorrea14/param_sweep.png')

    # plt.style.use('seaborn-darkgrid')
    # plt.rcParams.update({'font.size': 30})
    # plt.figure(figsize=(20,15))
    # best_curves = np.argsort(-np.array(aucs))[:NUM_MODELS_TO_KEEP]
    # for curve in best_curves:
    #     precision, recall, model = prs[curve]
    #     plt.plot(recall, precision, label=model, linewidth=5.0)
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.legend(frameon=True, prop={'size': 16})
    # plt.savefig('/home/chriscorrea14/param_sweep_truncated.png')
