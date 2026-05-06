#!/usr/bin/env python3
import sys
sys.path.insert(0, "/code/diffusha")  # so we can import diffusha

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from diffusha.data_collection.generate_data import ReplayBuffer
from util import SafeLimitsNormalizer


class BlockPushDataset(Dataset):
    def __init__(self, data_dirs, state_dim, action_dim, obs_horizon, pred_horizon):
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.state_dim = state_dim
        self.action_dim = action_dim

        all_obs = []
        all_actions = []

        for data_dir in data_dirs:
            print(f"loading {data_dir}...")
            rb = ReplayBuffer(data_dir, state_dim, action_dim)
            for fname, chunk in rb._file_cache.items():
                all_obs.append(chunk[:, :state_dim])
                all_actions.append(chunk[:, state_dim:state_dim + action_dim])

        self.obs = np.concatenate(all_obs, axis=0)         # (N, 7)
        self.actions = np.concatenate(all_actions, axis=0)  # (N, 2)
        print(f"total samples: {len(self.obs)}")

        # build normalizers from data
        self.normalizers = {
            "block_translation":     SafeLimitsNormalizer(torch.tensor(self.obs[:, 0:2])),
            "block_orientation":     SafeLimitsNormalizer(torch.tensor(self.obs[:, 2:3])),
            "ee_translation":        SafeLimitsNormalizer(torch.tensor(self.obs[:, 3:5])),
            "ee_target_translation": SafeLimitsNormalizer(torch.tensor(self.obs[:, 5:7])),
            "action":                SafeLimitsNormalizer(torch.tensor(self.actions)),
        }

    def __len__(self):
        return len(self.obs) - self.obs_horizon - self.pred_horizon

    def __getitem__(self, idx):
        obs_seq = self.obs[idx:idx + self.obs_horizon]       # (obs_horizon, 7)
        act_seq = self.actions[idx:idx + self.pred_horizon]  # (pred_horizon, 2)

        obs_tensor = torch.tensor(obs_seq, dtype=torch.float32)
        act_tensor = torch.tensor(act_seq, dtype=torch.float32)

        return {
            "block_translation":     self.normalizers["block_translation"].normalize(obs_tensor[:, 0:2]),
            "block_orientation":     self.normalizers["block_orientation"].normalize(obs_tensor[:, 2:3]),
            "ee_translation":        self.normalizers["ee_translation"].normalize(obs_tensor[:, 3:5]),
            "ee_target_translation": self.normalizers["ee_target_translation"].normalize(obs_tensor[:, 5:7]),
            "action":                self.normalizers["action"].normalize(act_tensor),
        }