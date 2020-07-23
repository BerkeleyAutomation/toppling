import os
import sys
import argparse
import trimesh
import numpy as np
from autolab_core import YamlConfig, RigidTransform
from toppling import TopplingDatasetModel


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='get topple matrices')
    parser.add_argument('obj_name', type=str, help="object name")
    parser.add_argument('--cfg', type=str, default='cfg/tools/toppling_graph.yaml', help='configuration file to use')
    args = parser.parse_args()

    # Create toppling model
    obj_name = args.obj_name
    cfg = YamlConfig(args.cfg)
    topple_model = TopplingDatasetModel(cfg["datasets"])
    
    # Load mesh and find stable poses
    obj_dir, obj_key = obj_name.split("~")
    obj_path = os.path.join('/nfs/diskstation/objects/meshes', obj_dir, obj_key + ".obj")
    mesh = trimesh.load(obj_path)
    stps, _ = mesh.compute_stable_poses(threshold=0.001)

    # Generate topple matrix and reassociate with stable poses
    topple_matrix, obj_poses = topple_model.topple_matrix(obj_name)
    if len(obj_poses) != len(stps):
        print(len(obj_poses), len(stps))
        sys.exit(1)
    op_zs = np.repeat(np.linalg.inv(obj_poses)[:, None, :3, 2], len(stps), axis=1)
    stp_zs = np.repeat(np.linalg.inv(stps)[None, :, :3, 2], len(obj_poses), axis=0)
    op_inds = np.linalg.norm(stp_zs - op_zs, axis=-1).argmin(axis=0)

    stp_topple_matrix = np.zeros_like(topple_matrix)
    for i, oi in enumerate(op_inds):
        stp_topple_matrix[oi, op_inds] = topple_matrix[i]
    
    np.save(os.path.join('/nfs/diskstation/projects/bandit_grasping/topple_matrices', obj_name), stp_topple_matrix)
    print(stp_topple_matrix)