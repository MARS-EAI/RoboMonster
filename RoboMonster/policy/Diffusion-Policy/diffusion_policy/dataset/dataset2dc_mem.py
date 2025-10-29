from typing import Union, Dict, Any, Tuple
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
from diffusion_policy.common.sampler import create_indices
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.common.normalize_util import get_image_range_normalizer, get_identity_normalizer_from_stat


class Dataset2D(Dataset):
    def __init__(self, 
                 dataset_path: str, 
                 horizon: int=1,
                 pad_before: int=0,
                 pad_after: int=0,
                 input_meta: Union[Dict[str, Any], None]=None,
                 seperate_action: bool=False,
                 episode_mask: Union[np.ndarray, None]=None
                 ) -> None:
        # self.data = h5py.File(dataset_path, "r")
        self.data = {}
        with h5py.File(dataset_path, 'r') as f:
            for key in f.keys():
                self.data[key] = f[key][:]
        episode_ends = self.data["episode_ends"]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)
        self.indices = create_indices(episode_ends, 
                sequence_length=horizon, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        self.horizon = horizon
        self.input_meta = input_meta
        self.separate_action = seperate_action

    def __len__(self):
        return len(self.indices)
    
    def padding(self, data: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
        """
        对数据进行填充，确保数据长度为指定的范围。
        :param data: 输入数据
        :param start_idx: 起始索引
        :param end_idx: 结束索引
        :return: 填充后的数据
        """
        if start_idx > 0:
            data[:start_idx] = data[start_idx]
        if end_idx < self.horizon:
            data[end_idx:] = data[end_idx - 1]

        return data
    
    def get_all_actions(self) -> Dict[str, np.ndarray]:
        res = {}
        for key in self.input_meta.keys():
            if key.startswith("action"):
                res[key] = self.data[key]
        
        if not self.separate_action:
            agent_num = len(res)
            action_list = []
            for i in range(agent_num):
                key = f"action_{i}"
                action_list.append(res[key])
                del res[key]
            res['action'] = np.concatenate(action_list, axis=-1)

        return res
    
    def get_normalizer(self, mode='limits', **kwargs):
        """
        获取数据归一化器，返回用于归一化动作和状态的归一化器。

        :param mode: 归一化模式（如 'limits'）。
        :param kwargs: 其他归一化器的参数。
        :return: 正常化器对象
        """
        # 从回放缓存中提取数据
        if self.separate_action:
            data = {}
            for key, value in self.get_all_actions().items():
                data[key] = value
                data[key.replace("action", "agent_pos")] = value
        else:
            actions = self.get_all_actions()['action']
            data = {
                'action': actions,
                'agent_pos': actions
            }
        # 创建并拟合归一化器
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # 为不同的相机图像获取归一化器
        for key in self.input_meta["obs"].keys():
            if key.startswith("head_cam"):
                normalizer[key] = get_image_range_normalizer()
            elif key.startswith("gaussian"):
                normalizer[key] = get_identity_normalizer_from_stat(
                    stat = {
                        'min': np.array([-1], dtype=np.float32),
                        'max': np.array([1], dtype=np.float32),
                        'mean': np.array([0], dtype=np.float32),
                        'std': np.array([1], dtype=np.float32)
                    }
                )
        return normalizer
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]
        res = {'obs': {}, 'sample_start_idx': np.array(sample_start_idx), 'buffer_start_idx': np.array(buffer_start_idx)}
        obs_keys = list(self.input_meta["obs"].keys())
        action_keys = [key for key in self.input_meta.keys() if key.startswith("action")]
        for key in obs_keys:
            if key.startswith("head_cam"):
                obs = self.data[key][buffer_start_idx:buffer_end_idx]
                obs = np.array(obs).astype(np.float32) / 255.0
            elif key.startswith("agent_pos"):
                obs = self.data[key.replace("agent_pos", "action")][buffer_start_idx:buffer_end_idx]
                obs = np.array(obs).astype(np.float32)
            else:
                obs = self.data[key][buffer_start_idx:buffer_end_idx]
                obs = np.array(obs).astype(np.float32)
            data = np.zeros((self.horizon, *obs.shape[1:]), dtype=np.float32)
            data[sample_start_idx:sample_end_idx] = obs
            data = self.padding(data, sample_start_idx, sample_end_idx)
            res['obs'][key] = data
        for key in action_keys:
            action = self.data[key][buffer_start_idx:buffer_end_idx]
            action = np.array(action).astype(np.float32)
            data = np.zeros((self.horizon, *action.shape[1:]), dtype=np.float32)
            data[sample_start_idx:sample_end_idx] = action
            data = self.padding(data, sample_start_idx, sample_end_idx)
            res[key] = data

        if not self.separate_action:
            agent_num = len(action_keys)
            agent_pos_list = []
            action_list = []
            for i in range(agent_num):
                key = f"agent_pos_{i}"
                agent_pos_list.append(res['obs'][key])
                del res['obs'][key]
                key = f"action_{i}"
                action_list.append(res[key])
                del res[key]
            res['obs']['agent_pos'] = np.concatenate(agent_pos_list, axis=-1)
            res['action'] = np.concatenate(action_list, axis=-1)

        return res


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    args = parser.parse_args()
    dataset = Dataset2D(args.dataset_path, horizon=8, 
                        input_meta={
                            "obs": {
                                "head_cam_0": [3, 256, 256],
                                "head_cam_1": [3, 256, 256],
                                "agent_pos_0": [8],
                                "agent_pos_1": [8],
                            },
                            "action_0": [8],
                            "action_1": [8],
                        },
                        seperate_action=False)
    norms = dataset.get_normalizer()
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        # print(data)
        break