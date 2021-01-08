from enum import auto, Enum

import MinkowskiEngine as ME
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import FeatureConfig, Mode
from data_loaders.sileane_loader import SileaneDataset


class DataKeys(Enum):
    PCD_COORDS = auto()
    PCD_FEATURES = auto()
    GT_COORDS = auto()
    GT_FEATURES = auto()
    NEG_INDICES = auto()
    POS_INDICES = auto()
    GT_TRANSFORMATION = auto()
    BATCH_LENGTH = auto()


def collate_pair_fn(list_data):
    coords0, coords1, feats0, feats1, pos_indices, neg_indices, trans = list(
        zip(*list_data))
    pos_indices_batch, neg_indices_batch, trans_batch, len_batch = [], [], [], []
    curr_start_inds = np.zeros((1, 2))

    def to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        else:
            raise ValueError(f'Can not convert to torch tensor, {x}')

    for batch_id, _ in enumerate(coords0):
        N0 = coords0[batch_id].shape[0]
        N1 = coords1[batch_id].shape[0]
        trans_batch.append(to_tensor(trans[batch_id]))

        pos_indices_batch.append(
            torch.from_numpy(np.array(pos_indices[batch_id]) + curr_start_inds))
        neg_indices_batch.append(
            torch.from_numpy(np.array(neg_indices[batch_id]) + curr_start_inds))
        len_batch.append([N0, N1])

        # Move the head
        curr_start_inds[0, 0] += N0
        curr_start_inds[0, 1] += N1

    coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)
    coords_batch1, feats_batch1 = ME.utils.sparse_collate(coords1, feats1)

    # Concatenate all lists
    trans_batch = torch.cat(trans_batch, 0).float()
    pos_indices_batch = torch.cat(pos_indices_batch, 0).int()
    neg_indices_batch = torch.cat(neg_indices_batch, 0).int()

    return {
        DataKeys.PCD_COORDS: coords_batch0,
        DataKeys.PCD_FEATURES: feats_batch0.float(),
        DataKeys.GT_COORDS: coords_batch1,
        DataKeys.GT_FEATURES: feats_batch1.float(),
        DataKeys.NEG_INDICES: neg_indices_batch,
        DataKeys.POS_INDICES: pos_indices_batch,
        DataKeys.GT_TRANSFORMATION: trans_batch,
        DataKeys.BATCH_LENGTH: len_batch
    }


def get_data_loaders(config: FeatureConfig):
    dataset = SileaneDataset(config)
    return {
        Mode.eval: DataLoader(dataset, num_workers=config.num_workers, batch_size=config.batch_size, shuffle=False,
                              collate_fn=collate_pair_fn, drop_last=True),
        Mode.train: DataLoader(dataset, num_workers=config.num_workers, batch_size=config.batch_size, shuffle=True,
                               collate_fn=collate_pair_fn, drop_last=True)
    }


if __name__ == "__main__":
    from config import FeatureConfig
    from time import time
    x = get_data_loaders(FeatureConfig())
    start = time()
    for data in x[Mode.eval]:
        print(data["pos_indices"].shape)
        print("ELLAPSED: {}".format(time() - start))
        start = time()
