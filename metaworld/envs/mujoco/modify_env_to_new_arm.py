from glob import glob
import re
import os
from tqdm import tqdm

import argparse

# SawyerXYZEnv => JacoEnv
# sawyer_xyz => jaco
# Sawyer => Jaco
# sawyer => jaco


def main(mujoco_dir, arm="ur5e", arm_upper="UR5e"):
    envs = glob(os.path.join(mujoco_dir, "sawyer_xyz", "v2", "*.py"))
    for env in tqdm(envs):
        with open(env, "r") as f:
            env_str = f.read()
        env_str = re.sub(
            r"SawyerXYZEnv",
            f"{arm_upper}Env",
            env_str,
        )
        env_str = re.sub(
            r"sawyer_xyz",
            f"{arm}",
            env_str,
        )
        env_str = re.sub(
            r"Sawyer",
            f"{arm_upper}",
            env_str,
        )
        env_str = re.sub(
            r"sawyer",
            f"{arm}",
            env_str,
        )

        new_file_name = os.path.basename(env).replace("sawyer", f"{arm}")
        new_env = os.path.join(mujoco_dir, f"{arm}", "v2", new_file_name)
        with open(new_env, "w") as f:
            f.write(env_str)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mujoco_dir", type=str, default="./")
    parser.add_argument("--arm", type=str, default="jaco")
    parser.add_argument("--arm_upper", type=str, default="Jaco")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(**vars(args))
