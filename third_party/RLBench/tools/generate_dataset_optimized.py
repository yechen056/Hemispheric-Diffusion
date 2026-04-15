#!/usr/bin/env python3
import sys
import os
import pathlib
import logging
import numpy as np
import zarr
import numcodecs
import rich_click as click
from rich.logging import RichHandler

# 引入项目路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = str(pathlib.Path(CURRENT_DIR).parent)
sys.path.append(ROOT_DIR)

# RLBench & PyRep
from rlbench import ObservationConfig
from rlbench.observation_config import CameraConfig
from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
from rlbench.action_modes.arm_action_modes import BimanualJointPosition
from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment
from rlbench.backend.task import BIMANUAL_TASKS_PATH
from rlbench.backend.exceptions import BoundaryError, WaypointError, InvalidActionError

# Modular Policy
from modular_policy.common.replay_buffer import ReplayBuffer

# ---------------- 配置 ----------------
CAMERA_NAMES = ["front", "overhead", "wrist_left", "wrist_right", "over_shoulder_left", "over_shoulder_right"]

def get_bimanual_state(obs):
    """从 RLBench Observation 提取 16维 状态 (14关节 + 2夹爪)"""
    try:
        left = obs.left
        right = obs.right
        j = np.concatenate([left.joint_positions, right.joint_positions]) # 14
        g = np.array([left.gripper_open, right.gripper_open], dtype=np.float32) # 2
        return np.concatenate([j, g]) # 16
    except Exception as e:
        logging.error(f"Error extracting state: {e}")
        return np.zeros(16, dtype=np.float32)

def extract_episode_data(demo):
    """
    将 RLBench 的 demo 对象转换为 ReplayBuffer 接受的字典格式
    """
    num_steps = len(demo)
    ep_data = {
        'joint_positions': [], 
        'gripper_open': [], 
        'action': [],
        # 兼容性: 同时保存合并后的 robot_joint
        'robot_joint': [] 
    }
    
    # 初始化相机列表
    for cam in CAMERA_NAMES:
        ep_data[f'{cam}_rgb'] = []
        ep_data[f'{cam}_depth'] = []
        ep_data[f'{cam}_extrinsics'] = []
        ep_data[f'{cam}_intrinsics'] = []

    for i in range(num_steps):
        obs = demo[i]
        
        # 1. 状态提取
        full_state = get_bimanual_state(obs) # 16维
        joints = full_state[:14]
        gripper = full_state[14:]
        
        ep_data['joint_positions'].append(joints)
        ep_data['gripper_open'].append(gripper)
        ep_data['robot_joint'].append(full_state) # 16维合并状态

        # 2. Action 对齐 (Action_t = State_t+1)
        if i < num_steps - 1:
            next_state = get_bimanual_state(demo[i+1])
            ep_data['action'].append(next_state)
        else:
            ep_data['action'].append(full_state) # 最后一步保持

        # 3. 视觉数据提取
        for cam in CAMERA_NAMES:
            # RGB (H, W, 3)
            rgb = obs.perception_data[f'{cam}_rgb']
            ep_data[f'{cam}_rgb'].append(rgb)
            
            # Depth (H, W) - Float32 (Meters)
            # 关键：RLBench 直接返回 float32，无需解码
            depth = obs.perception_data[f'{cam}_depth']
            ep_data[f'{cam}_depth'].append(depth)
            
            # Matrices
            misc = obs.misc
            ep_data[f'{cam}_extrinsics'].append(misc[f'{cam}_camera_extrinsics'])
            ep_data[f'{cam}_intrinsics'].append(misc[f'{cam}_camera_intrinsics'])

    # 转换为 Numpy 数组
    numpy_data = {}
    for k, v in ep_data.items():
        arr = np.array(v)
        if 'rgb' in k:
            numpy_data[k] = arr.astype(np.uint8)
        else:
            numpy_data[k] = arr.astype(np.float32)
            
    return numpy_data

def get_bimanual_tasks():
    tasks = [t.replace('.py', '') for t in os.listdir(BIMANUAL_TASKS_PATH) 
             if t != '__init__.py' and t.endswith('.py')]
    return sorted(tasks)

# ---------------- CLI ----------------
@click.command()
@click.option("--save_path", default="data/bimanual_generated.zarr", help="Output Zarr file path")
@click.option('--tasks', type=click.Choice(get_bimanual_tasks()), multiple=True, default=['bimanual_pick_plate'])
@click.option("--episodes", default=10, help="Number of episodes to generate")
@click.option("--headless/--no-headless", default=True, is_flag=True)
@click.option('--image-size', type=click.Choice(["128x128", "256x256"]), default="128x128")
def main(save_path, tasks, episodes, headless, image_size):
    # 1. 设置日志
    logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])
    img_w, img_h = map(int, image_size.split("x"))

    # 2. 准备 Zarr ReplayBuffer
    if os.path.exists(save_path):
        if click.confirm(f"'{save_path}' already exists. Overwrite?", default=False):
            import shutil
            shutil.rmtree(save_path)
        else:
            # 追加模式 (Append Mode)
            logging.info(f"Appending to existing dataset: {save_path}")

    # 创建 Zarr 存储后端
    store = zarr.DirectoryStore(save_path)
    root = zarr.group(store=store)
    
    # 初始化 ReplayBuffer (如果是新文件，它会创建结构；如果是旧文件，它会读取 meta)
    replay_buffer = ReplayBuffer.create_from_group(root)
    
    # 设置压缩器 (图像 shuffle, 其他数据 bitshuffle)
    img_compressor = numcodecs.Blosc(cname='zstd', clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
    data_compressor = numcodecs.Blosc(cname='zstd', clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    
    compressors_dict = {}
    # 我们不仅要设置压缩，还需要告知 ReplayBuffer 哪些 key 用什么压缩
    # 但 ReplayBuffer.add_episode 默认不接受 dict，我们可以利用其内部逻辑
    # 最好的方法是让 ReplayBuffer 自动处理 chunking，我们只指定 compressor
    # 这里为了简单，我们在第一次 add_episode 时传入默认 compressor，ReplayBuffer 会记住

    # 3. 初始化 RLBench 环境
    logging.info("Initializing Simulation...")
    task_class = task_file_to_task_class(tasks[0], True) # 假设只跑一个任务

    obs_config = ObservationConfig()
    obs_config.set_all(True) # 开启所有数据(含触觉)
    
    # 关键配置
    default_config_params = {
        "image_size": [img_w, img_h], 
        "depth_in_meters": True,  # <--- 重点：直接获取 Float32 深度
        "masks_as_one_channel": False
    }
    camera_configs = {name: CameraConfig(**default_config_params) for name in CAMERA_NAMES}
    obs_config.camera_configs = camera_configs

    rlbench_env = Environment(
        action_mode=BimanualMoveArmThenGripper(BimanualJointPosition(), BimanualDiscrete()),
        obs_config=obs_config,
        robot_setup='dual_panda',
        headless=headless
    )
    rlbench_env.launch()

    task_env = rlbench_env.get_task(task_class)
    possible_variations = task_env.variation_count()

    # 4. 生成循环
    current_episodes = replay_buffer.n_episodes
    target_episodes = current_episodes + episodes
    
    logging.info(f"Starting generation. Current: {current_episodes}, Target: {target_episodes}")

    while replay_buffer.n_episodes < target_episodes:
        ep_idx = replay_buffer.n_episodes
        attempts = 10
        success = False

        while attempts > 0:
            try:
                variation = np.random.randint(possible_variations)
                task_env.set_variation(variation)
                descriptions, obs = task_env.reset()
                
                # 获取演示数据
                demo, = task_env.get_demos(amount=1, live_demos=True)
                
                # 提取数据
                ep_data = extract_episode_data(demo)
                
                # 写入 ReplayBuffer (直接写入磁盘)
                # 为图像数据指定特定的压缩器
                custom_compressors = {}
                for k in ep_data.keys():
                    if 'rgb' in k: custom_compressors[k] = img_compressor
                    else: custom_compressors[k] = data_compressor

                replay_buffer.add_episode(ep_data, compressors=custom_compressors)
                
                logging.info(f"✅ Saved Episode {ep_idx} (Len: {len(demo)})")
                success = True
                break

            except (BoundaryError, WaypointError, InvalidActionError) as e:
                attempts -= 1
                # logging.warning(f"Retry {10-attempts}: {e}")
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                attempts -= 1
        
        if not success:
            logging.error(f"❌ Failed to generate episode {ep_idx} after 10 attempts.")
            # 可以选择 continue 重试，或者 break
    
    rlbench_env.shutdown()
    logging.info(f"🎉 Done! Dataset saved to {save_path}")

if __name__ == '__main__':
    main()