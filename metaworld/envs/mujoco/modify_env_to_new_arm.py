from glob import glob
import re
import os
from tqdm import tqdm

# SawyerXYZEnv => JacoEnv
# sawyer_xyz => jaco
# Sawyer => Jaco
# sawyer => jaco


def main(mujoco_dir, arm="jaco"):
    envs = glob(os.path.join(mujoco_dir, "sawyer_xyz", "v2", "*.py"))
    arm_upper = f"{arm[0].upper()}{arm[1:]}"
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


if __name__ == "__main__":
    mujoco_dir = "./"
    main(mujoco_dir)
