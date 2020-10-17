import MinkowskiEngine as ME
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config, Mode
from data_loaders.sileane_loader import SileaneDataset


def collate_pair_fn(list_data):
    xyz0, xyz1, coords0, coords1, feats0, feats1, matching_inds, trans = list(
        zip(*list_data))
    xyz_batch0, xyz_batch1 = [], []
    matching_inds_batch, trans_batch, len_batch = [], [], []

    batch_id = 0
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

        xyz_batch0.append(to_tensor(xyz0[batch_id]))
        xyz_batch1.append(to_tensor(xyz1[batch_id]))

        trans_batch.append(to_tensor(trans[batch_id]))

        matching_inds_batch.append(
            torch.from_numpy(np.array(matching_inds[batch_id]) + curr_start_inds))
        len_batch.append([N0, N1])

        # Move the head
        curr_start_inds[0, 0] += N0
        curr_start_inds[0, 1] += N1

    coords_batch0, feats_batch0 = ME.utils.sparse_collate(coords0, feats0)
    coords_batch1, feats_batch1 = ME.utils.sparse_collate(coords1, feats1)

    # Concatenate all lists
    xyz_batch0 = torch.cat(xyz_batch0, 0).float()
    xyz_batch1 = torch.cat(xyz_batch1, 0).float()
    trans_batch = torch.cat(trans_batch, 0).float()
    matching_inds_batch = torch.cat(matching_inds_batch, 0).int()

    return {
        'pcd0': xyz_batch0,
        'pcd1': xyz_batch1,
        'sinput0_C': coords_batch0,
        'sinput0_F': feats_batch0.float(),
        'sinput1_C': coords_batch1,
        'sinput1_F': feats_batch1.float(),
        'correspondences': matching_inds_batch,
        'T_gt': trans_batch,
        'len_batch': len_batch
    }


def get_data_loaders(config: Config):
    dataset = SileaneDataset(config)
    return {
        Mode.eval: DataLoader(dataset, num_workers=config.num_workers, batch_size=config.batch_size, shuffle=False,
                              collate_fn=collate_pair_fn, drop_last=True),
        Mode.train: DataLoader(dataset, num_workers=config.num_workers, batch_size=config.batch_size, shuffle=True,
                               collate_fn=collate_pair_fn, drop_last=True)
    }


if __name__ == "__main__":
    from config import FeatureConfig
    x = get_data_loaders(FeatureConfig())
    print(x)
