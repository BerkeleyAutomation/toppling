import os
import argparse
import trimesh
import numpy as np
import tqdm

from autolab_core import YamlConfig
from dexnet.grasping import GraspableObject3D
from toppling.model import TopplingModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_path", type=str, help="path to mesh")
    parser.add_argument("--cfg", type=str, default="cfg/tools/topple_model.yaml")
    args = parser.parse_args()

    mesh_path = args.mesh_path
    mesh_name = os.path.dirname(mesh_path).split("/")[-1] + "~" + os.path.splitext(os.path.basename(mesh_path))[0]
    mesh = trimesh.load(mesh_path).apply_scale(0.001)
    mesh.fix_normals()

    cfg = YamlConfig(args.cfg)
    stps, probs = mesh.compute_stable_poses(threshold=0.001)
    topple_mat = np.zeros((len(stps), len(stps)))
    for i, stp in enumerate(tqdm.tqdm(stps)):
        tf_mesh = mesh.copy().apply_transform(stp)
        obj = GraspableObject3D(tf_mesh)
        model = TopplingModel(cfg, obj=obj)

        # Sample points on mesh
        vertices, face_ind = trimesh.sample.sample_surface_even(tf_mesh, 100)
        
        # Cut out points that are too close to the ground
        valid_vertex_ind = vertices[:, 2] > 0.01
        vertices, face_ind = vertices[valid_vertex_ind], face_ind[valid_vertex_ind]

        normals = tf_mesh.face_normals[face_ind]
        push_directions = -normals.copy()
        poses, vertex_probs, min_required_forces = model.predict(
            vertices, 
            normals, 
            push_directions, 
            use_sensitivity=True
        )
        final_poses = np.array([p.matrix for p in poses]).dot(stp)
        pose_inds = np.linalg.norm(final_poses[:, None] - stps[None, ...], axis=(2,3)).argmin(axis=-1)
        uniq_inds, uniq_inv = np.unique(pose_inds, return_inverse=True)
        topple_probs = np.zeros(len(uniq_inds))
        np.add.at(topple_probs, uniq_inv, vertex_probs.mean(axis=0))
        topple_mat[i, uniq_inds] = topple_probs
        
    np.save(os.path.join("/nfs/diskstation/projects/bandit_grasping/topple_matrices", mesh_name + ".npy"), topple_mat)
