import numpy as np

from toppling.models import TopplingModel

def get_model_config(config, ground_friction_coeff, finger_friction_coeff, finger_sigma, push_direction_sigma, baseline):
    config['ground_friction_coeff'] = ground_friction_coeff
    config['finger_friction_coeff'] = finger_friction_coeff
    config['finger_sigma'] = finger_sigma
    config['push_direction_sigma'] = push_direction_sigma
    config['baseline'] = baseline
    return config

def tmp(env, datasets, obj_id_to_keys, models, model_names, use_sensitivities):
    for dataset, obj_id_to_key in zip(datasets, obj_id_to_keys):
        for i in range(dataset.num_datapoints):
            datapoint = dataset.datapoint(i)

            dataset_name, key = obj_id_to_key[str(datapoint['obj_id'])].split(KEY_SEP_TOKEN)
            obj = env._state_space._database.dataset(dataset_name)[key]
            orig_pose = to_rigid(datapoint['obj_pose'])
            obj.T_obj_world = orig_pose
            env.state.obj = obj

            for model, model_name, use_sensitivity in zip(models, model_names, use_sensitivities):
                model.load_object(env.state)
                _, vertex_probs, _ = model.predict(
                    [datapoint['vertex']],
                    [datapoint['normal']],
                    [-datapoint['normal']], # push dir
                    use_sensitivity=use_sensitivity
                )
                print model_name, 1-vertex_probs[0],
            print ''
            env.render_3d_scene()
            start_point = datapoint['vertex'] + .06*datapoint['normal']
            end_point = datapoint['vertex']
            h1 = np.array([[1,0,0],[0,0.7071,-0.7071],[0,0.7071,0.7071]]).dot(-normal)
            h2 = np.array([[1,0,0],[0,0.7071,0.7071],[0,-0.7071,0.7071]]).dot(-normal)
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
    args = parser.parse_args()
    config = YamlConfig(args.config_filename)
    if config['debug']:
        random.seed(SEED)
        np.random.seed(SEED)

    models = [
        TopplingModel(get_model_config(config, )),
        TopplingModel(get_model_config(config, )),
        TopplingModel(get_model_config(config, )),
        TopplingModel(get_model_config(config, )),
        TopplingModel(get_model_config(config, ))
    ]
