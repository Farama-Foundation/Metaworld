import time

from metaworld import MT1
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS
import importlib
from gymnasium.wrappers.record_video import RecordVideo
import numpy as np
import pickle
from PIL import Image
envs = list(ALL_V2_ENVIRONMENTS.keys())
'''
envs = ['bin-picking-v2']
f = open('raw-captions(1).pkl', 'rb')
caption_data = pickle.load(f)
for env in envs:
    print(env + ' ', caption_data.get('success_videos__' + env + '_1', [[]])[0])

exit(0)'''

for e in ['drawer-close-v2']:
    parts = e.split('-')
    new_parts = []
    for i in range(len(parts)):
        new_parts.append(parts[i][0].upper() + parts[i][1:])
    p_name = 'Sawyer' + ''.join(new_parts) + 'Policy'
    if e != 'peg-insert-side-v2':
        policy_class = importlib.import_module(f'metaworld.policies.sawyer_{e.replace("-", "_")}_policy')
        policy = getattr(policy_class, p_name)
    else:
        from metaworld.policies.sawyer_peg_insertion_side_v2_policy import SawyerPegInsertionSideV2Policy as policy
    p = policy()
    mt1 = MT1(e, seed=42)
    task_num = 0
    current_results = []
    alpha = 0.8
    noisy = False
    num_steps = 500
    if noisy:
        folder = f'videos//{e}/'
    else:
        folder = f'videos/success_videos/{e}/'
    task_rewards = 0.0
    task_success = 0
    grasp_act = None
    for idx, task in enumerate(mt1.train_tasks):
        env = mt1.train_classes[e](render_mode='rgb_array', ) # reward_func_version='v1'
        env.set_task(task)
        obs, info = env.reset()
        env = RecordVideo(env, video_folder=folder, name_prefix=e + ' ' + str(task_num), video_length=1000)
        info['success'] = 0.0
        count = 0
        success_count = 0
        success_reward = 0.0
        first_grasp = None
        while count < num_steps:
            a = p.get_action(obs)
            next_state, reward, terminate, truncate, info = env.step(a)
            success_reward += reward
            obs = next_state
            count += 1
            if int(info['success']) == 1:
                break
        print('after loop')
        if env:
            env.close()
        current_results.append(success_count)
        task_num += 1
        if success_count >= 1:
            task_rewards += success_reward
            task_success += 1
        print(success_reward)
    if task_success > 0:
        print(e, float(task_rewards) / task_success, task_rewards, task_success)
    else:
        print(e, ' no successes for this env')

    #with open(folder + 'successes.txt', 'w') as f:
    #    for s in current_results:
    #        f.writelines(str(int(s > 1)) + '\n')


