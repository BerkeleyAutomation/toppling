import argparse
import numpy as np
import os

from autolab_core import Point, RigidTransform, YamlConfig, TensorDataset
from dexnet.constants import *
from dexnet.envs import GraspingEnv
from dexnet.visualization import DexNetVisualizer3D as vis3d
from toppling.models import TopplingModel
from toppling import camera_pose

def to_rigid(mat):
    rot, trans = RigidTransform.rotation_and_translation_from_matrix(mat)
    return RigidTransform(rot, trans, 'obj', 'world')

def parse_args():
    default_config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       '..',
                                       'cfg/tools/benchmark_topple_policy_graspingenv.yaml'
    )
    parser = argparse.ArgumentParser(description='Rollout a policy for bin picking in order to evaluate performance')
    parser.add_argument('--config_filename', type=str, default=default_config_filename, help='configuration file to use')
    parser.add_argument('--dataset', type=str, help='Which dataset to load toppling experiments from')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = YamlConfig(args.config_filename)
    if config['debug']:
        random.seed(SEED)
        np.random.seed(SEED)

    model = TopplingModel(config['model'])
    dataset = TensorDataset.open(args.dataset)
    env = GraspingEnv(config, config['vis'])

    for model in models:
        for i in range(dataset.num_datapoints):
            datapoint = dataset.datapoint(i)

            dataset_name, key = self.obj_keys[datapoint['obj_id']].split(KEY_SEP_TOKEN)
            obj = env._state_space._database.dataset(dataset_name)[key]
            obj.T_obj_world = to_rigid(datapoint['obj_pose'])
            env.state.obj = obj
            
            model.load_object(env.state)
            model.predict()