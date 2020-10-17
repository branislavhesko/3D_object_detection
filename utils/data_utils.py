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


def find_negative_matches(coords_gt, coords_object, positive_distance_limit):
    distance = torch.sqrt(torch.pow(coords_gt - coords_object, 2).sum(-1))
    return coords_gt[distance > positive_distance_limit, ...], coords_object[distance > positive_distance_limit, ...]


def find_model_opposite_points(coords_gt, coords_object, positive_distance_limit):
    distance = torch.sqrt(torch.pow(coords_gt.unsqueeze(0) - coords_gt.unsqueeze(1), 2).sum(-1))
    random_distribution = torch.distributions.exponential.Exponential(rate=5.)
    distance *= (1 - random_distribution.sample(sample_shape=distance.shape))
    furthest_points = torch.argmax(distance, dim=1)
    return coords_gt, find_positive_matches(coords_gt[furthest_points], coords_object, positive_distance_limit)[1]


def find_positive_matches(coords_gt, coords_object, positive_distance_limit):
    distance = torch.sqrt(torch.pow(coords_gt.unsqueeze(0) - coords_object.unsqueeze(1), 2).sum(-1))
    closest_points = torch.argmin(distance, dim=1)
    picked_gts = torch.arange(coords_gt.shape[0])
    correct_idx = torch.where((distance[picked_gts, closest_points] < positive_distance_limit))
    return coords_gt[picked_gts[correct_idx]], coords_object[closest_points[correct_idx]]


def distance(feats1, feats2, type_="L2"):
    coef = 1 if type_ == "L1" else 2
    return torch.sqrt(torch.pow(feats1 - feats2, coef).sum(-1) + 1e-7)


def pca(tensor, n_components=3):
    assert tensor.shape[1] < tensor.shape[0]
    cov_mat = torch.matmul(tensor.transpose(0, 1), tensor) / (tensor.shape[0] - 1)
    eigen = torch.eig(cov_mat, eigenvectors=True)
    return torch.matmul(tensor, eigen[1][:, :n_components])