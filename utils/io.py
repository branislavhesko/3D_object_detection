import numpy as np


def load_model(model_path):
    pass


def make_transformation_matrix(vectors_dict):
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(vectors_dict["R"]).reshape((3, 3))
    matrix[:3, -1] = vectors_dict["t"]
    return matrix

