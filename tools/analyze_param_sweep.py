import numpy as np
import os
import argparse
import json

from toppling.models import TopplingModel
from toppling import is_equivalent_pose
from autolab_core import YamlConfig, TensorDataset, RigidTransform
from dexnet.envs import GraspingEnv
from dexnet.constants import *
from dexnet.visualization import DexNetVisualizer3D as vis3d

NUM_PER_DATAPOINT = 10

def to_rigid(mat):
    rot, trans = RigidTransform.rotation_and_translation_from_matrix(mat)
    return RigidTransform(rot, trans, 'obj', 'world')

def get_model_config(config, ground_friction_coeff, finger_friction_coeff, finger_sigma, push_direction_sigma, baseline):
    config['ground_friction_coeff'] = ground_friction_coeff
    config['finger_friction_coeff'] = finger_friction_coeff
    config['finger_sigma'] = finger_sigma
    config['push_direction_sigma'] = push_direction_sigma
    config['baseline'] = baseline
    return config

def visualize(env, datasets, obj_id_to_keys, models, model_names, use_sensitivities):
    for dataset, obj_id_to_key in zip(datasets, obj_id_to_keys):
        for i in range(dataset.num_datapoints):
            datapoint = dataset.datapoint(i)

            dataset_name, key = obj_id_to_key[str(datapoint['obj_id'])].split(KEY_SEP_TOKEN)
            obj = env._state_space._database.dataset(dataset_name)[key]
            orig_pose = to_rigid(datapoint['obj_pose'])
            obj.T_obj_world = orig_pose
            env.state.obj = obj

            actual = 0
            for i in range(NUM_PER_DATAPOINT):
                mat = datapoint['actual_poses'][i*4:(i+1)*4]
                actual_pose = to_rigid(mat)
                if not is_equivalent_pose(actual_pose, orig_pose):
                    actual += 1
            actual /= float(NUM_PER_DATAPOINT)
            
            probs = []
            for model, model_name, use_sensitivity in zip(models, model_names, use_sensitivities):
                model.load_object(env.state)
                _, vertex_probs, _ = model.predict(
                    [datapoint['vertex']],
                    [datapoint['normal']],
                    [-datapoint['normal']], # push dir
                    use_sensitivity=use_sensitivity
                )
                probs.append(1-vertex_probs[0,0])

            if (abs(probs[3] - actual) - abs(probs[2] - actual)) > .2:
                print 'actual {} {} {} {} {} {} {} {} {}'.format(actual,
                    model_names[0], probs[0], model_names[1], probs[1],
                    model_names[2], probs[2], model_names[3], probs[3]
                )
                env.render_3d_scene()
                start_point = datapoint['vertex'] + .06*datapoint['normal']
                end_point = datapoint['vertex']
                shaft_points = [start_point, end_point]
                h1 = np.array([[0.7071,-0.7071,0],[0.7071,0.7071,0],[0,0,1]]).dot(-datapoint['normal'])
                h2 = np.array([[0.7071,0.7071,0],[-0.7071,0.7071,0],[0,0,1]]).dot(-datapoint['normal'])
                head_points = [end_point - 0.02*h2, end_point, end_point - 0.02*h1]
                vis3d.plot3d(shaft_points, color=[1,0,0], tube_radius=.002)
                vis3d.plot3d(head_points, color=[1,0,0], tube_radius=.002)
                vis3d.show()

if __name__ == '__main__':
    # with open('/nfs/diskstation/db/toppling/param_sweep.log', 'r') as file:
    #     lines, tvs = [], []
    #     for i, line in enumerate(file):
    #         # if i > 126 or not line.startswith('ground friction'):
    #         if i < 126 or not line.startswith('ground friction'):
    #             continue
    #         combined_tv = line.split('Combined TV: ')[1]
    #         combined_tv = combined_tv.split(' ')[0]
    #         lines.append(line)
    #         tvs.append(combined_tv)
    #     best_combined_tvs = np.argsort(tvs)[:10]
    #     print best_combined_tvs
    #     for i in best_combined_tvs:
    #         print lines[i]
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/benchmark_topple_policy_graspingenv.yaml'
    )
    parser = argparse.ArgumentParser(description='Rollout a policy for bin picking in order to evaluate performance')
    parser.add_argument('--config_filename', type=str, default=default_config_filename, help='configuration file to use')
    parser.add_argument('--datasets', type=str, default='/nfs/diskstation/db/toppling/experiments/', help='Which dataset to load toppling experiments from')
    args = parser.parse_args()
    config = YamlConfig(args.config_filename)
    if config['debug']:
        random.seed(SEED)
        np.random.seed(SEED)

    model_config = config['model']
    models = [
        TopplingModel(get_model_config(model_config, 2.7351e-01, 8.0404e-01, 7.4000e-04, 7.0320e-02, True)),
        TopplingModel(get_model_config(model_config, 3.6747e-01, 9.9801e-01, 8.3000e-04, 8.9320e-02, False)),
        TopplingModel(get_model_config(model_config, 3.4084e-01, 6.6242e-01, 6.3000e-04, 8.8170e-02, True)),
        TopplingModel(get_model_config(model_config, 3.1186e-01, 9.3045e-01, 4.1000e-04, 8.3230e-02, False)),
        #TopplingModel(get_model_config(config, ))
    ]
    use_sensitivities = [False, False, True, True]
    model_names = ['Baseline', 'Baseline+Rotations', 'Baseline+Robustness', 'Robust Model']

    env = GraspingEnv(config, config['vis'])
    env.reset()

    datasets, obj_id_to_keys = [], []
    for dataset_name in os.listdir(args.datasets):
        dataset_name = dataset_name.split(' ')[0]
        dataset_path = os.path.join(args.datasets, dataset_name)
        datasets.append(TensorDataset.open(dataset_path))
        with open(os.path.join(dataset_path, "obj_keys.json"), "r") as read_file:
            obj_id_to_keys.append(json.load(read_file))
    visualize(env, datasets, obj_id_to_keys, models, model_names, use_sensitivities)
