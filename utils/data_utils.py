import os

import numpy as np
import open3d
import torch


def load_model(model_path):
    extension = os.path.splitext(model_path)[1][1:]
    if extension == "ply":
        pcd = open3d.io.read_point_cloud(model_path)
        return torch.from_numpy(np.array(pcd.points)).float()


def make_transformation_matrix(vectors_dict):
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(vectors_dict["R"]).reshape((3, 3))
    matrix[:3, -1] = vectors_dict["t"]
    return matrix


if __name__ == "__main__":
    load_model("./data/sileane/gear/mesh.ply")
