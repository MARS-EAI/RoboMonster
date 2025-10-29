import os
from typing import Union
from tqdm import tqdm
import h5py
import numpy as np
from torch.utils.data import Dataset
import argparse

from mani_skill.utils.io_utils import load_json
from mani_skill.utils import common

# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out

class ManiSkillTrajectoryDataset(Dataset):
    """
    A general torch Dataset you can drop in and use immediately with just about any trajectory .h5 data generated from ManiSkill.
    This class simply is a simple starter code to load trajectory data easily, but does not do any data transformation or anything
    advanced. We recommend you to copy this code directly and modify it for more advanced use cases

    Args:
        dataset_file (str): path to the .h5 file containing the data you want to load
        load_count (int): the number of trajectories from the dataset to load into memory. If -1, will load all into memory
        success_only (bool): whether to skip trajectories that are not successful in the end. Default is false
        device: The location to save data to. If None will store as numpy (the default), otherwise will move data to that device
    """

    def __init__(self, dataset_file: str, load_count=-1, success_only: bool = False, device = None) -> None:
        self.dataset_file = dataset_file
        self.device = device
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.obs = []
        self.actions = []
        self.terminated = []
        self.truncated = []
        self.success, self.fail, self.rewards = None, None, None
        if load_count == -1:
            load_count = len(self.episodes)

        self.load_count = load_count
        self.success_only = success_only

    def __len__(self):
        return self.load_count

    def __getitem__(self, eps_id):
        eps = self.episodes[eps_id]
        if self.success_only: 
            assert "success" in eps, "episodes in this dataset do not have the success attribute, cannot load dataset with success_only=True"
            if not eps["success"]:
                return None

        trajectory = self.data[f"traj_{eps['episode_id']}"]
        trajectory = load_h5_data(trajectory)

        eps_len = int(eps['elapsed_steps'])

        # exclude the final observation as most learning workflows do not use it
        obs = common.index_dict_array(trajectory["obs"], slice(eps_len))
        
        obs = obs["sensor_data"]
        action = trajectory["actions"]
        # terminated = trajectory["terminated"]
        # truncated = trajectory["truncated"]
        res = dict(
            obs=obs,
            action=action,
            # terminated=terminated,
            # truncated=truncated,
        )

        return res
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __del__(self):
        self.data.close()


def main(dataset_path: str, output_path: str, load_num: int, agent_num: int) -> None:
    dataset = ManiSkillTrajectoryDataset(dataset_file=dataset_path, load_count=load_num)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    comp_kwaegs = {'compression': 'gzip', 'compression_opts': 4}
    episode_ends = []
    end = 0
    with h5py.File(output_path, "w") as f:
        for i, data in tqdm(enumerate(dataset), desc="Loading data", total=len(dataset)):
            if data is None:
                continue
            obs = data["obs"]
            action = data["action"]

            if agent_num > 1:
                camera_name = "head_camera_global"
            else:
                camera_name = "head_camera"

            if camera_name not in obs:
                print(f"Warning: Camera '{camera_name}' not found in trajectory {i}. Skipping.")
                continue

            is_multi_agent_action_format = isinstance(action, dict)
            
            if is_multi_agent_action_format:
                action_len = len(action['panda-0'])
            else:
                action_len = len(action)
            
            obs_len = len(obs[camera_name]["rgb"])
            min_len = min(action_len, obs_len)

            if action_len != obs_len:
                print(f"Trajectory {i}: Length mismatch. Action: {action_len}, Obs: {obs_len}. Using min: {min_len}.")

            head_cam = obs[camera_name]["rgb"][:min_len]
            head_cam = np.array(head_cam).astype(np.uint8)
            head_cam = np.moveaxis(head_cam, -1, -3) # from (L, H, W, C) to (L, C, H, W)

            if i == 0:
                f.create_dataset(
                    "head_cam",
                    data=head_cam,
                    shape=head_cam.shape,
                    maxshape=(None, *head_cam.shape[1:]),
                    dtype="uint8",
                    **comp_kwaegs
                )
            else:
                f["head_cam"].resize((f["head_cam"].shape[0] + head_cam.shape[0]), axis=0)
                f["head_cam"][-head_cam.shape[0]:] = head_cam

            for agent_id in range(agent_num):
                if is_multi_agent_action_format:
                    agent_action = action[f'panda-{agent_id}'][:min_len]
                else:
                    if agent_id != 0: break 
                    agent_action = action[:min_len]
                
                agent_action = np.array(agent_action).astype(np.float32)
                action_dset_name = f"action_{agent_id}"

                if i == 0:
                    f.create_dataset(
                        action_dset_name,
                        data=agent_action,
                        shape=agent_action.shape,
                        maxshape=(None, *agent_action.shape[1:]),
                        dtype="float32",
                        **comp_kwaegs
                    )
                else:
                    if action_dset_name not in f:
                        print(f"Error: Dataset {action_dset_name} not created in the first trajectory.")
                        continue
                    f[action_dset_name].resize((f[action_dset_name].shape[0] + agent_action.shape[0]), axis=0)
                    f[action_dset_name][-agent_action.shape[0]:] = agent_action

            end += min_len
            episode_ends.append(end)

        f.create_dataset(
            "episode_ends",
            data=np.array(episode_ends),
            **comp_kwaegs
        )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output")
    parser.add_argument("--agent_num", type=int, required=True, help="Number of agents")
    parser.add_argument("--load_num", type=int, required=True, help="Number of trajectories to load")
    args = parser.parse_args()

    main(args.dataset_path, args.output_path, args.load_num, args.agent_num)