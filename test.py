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


'''
import re

def extract_numbers(text):
    # Using a regular expression to find all occurrences of the pattern
    numbers = re.findall(r'\w+-v2 (\d+\.\d+)', text)
    return numbers

v2 = """assembly-v2 4295.877104694405 214793.85523472022 50
basketball-v2 4074.675908864069 203733.79544320345 50
bin-picking-v2 3793.557086152024 189677.8543076012 50
box-close-v2 3190.2513806720435 153132.0662722581 48
button-press-topdown-v2 3658.034173893188 182901.7086946594 50
button-press-topdown-wall-v2 2.1814562201181356 109.07281100590679 50
button-press-v2 557.2289823205454 27861.449116027266 50
button-press-wall-v2 2906.7731337862206 142431.8835555248 49
coffee-button-v2 502.0791243757947 25103.956218789735 50
coffee-pull-v2 4201.198302424607 210059.91512123038 50
coffee-push-v2 781.5452731299629 39077.263656498144 50
dial-turn-v2 3783.657604762353 189182.88023811765 50
disassemble-v2 3929.333570822413 196466.67854112067 50
door-close-v2 4486.8303014051435 224341.51507025716 50
door-lock-v2 3276.632413116681 163831.62065583406 50
door-open-v2 4462.204559137688 223110.2279568844 50
door-unlock-v2 3675.2551455808753 183762.75727904376 50
hand-insert-v2 4373.331826530643 218666.5913265322 50
drawer-close-v2 4231.674426011076 211583.72130055382 50
drawer-open-v2 4043.4684949445586 202173.42474722792 50
faucet-open-v2 4149.168471205682 207458.4235602841 50
faucet-close-v2 4091.195746293934 204559.7873146967 50
hammer-v2 1276.0828023960428 63804.140119802134 50
handle-press-side-v2 969.3445927132926 48467.22963566463 50
handle-press-v2 1259.357391820331 62967.869591016555 50
handle-pull-side-v2 3219.7246207429666 160986.23103714833 50
handle-pull-v2 3946.7587048766964 197337.93524383483 50
lever-pull-v2 894.8887270326768 44744.43635163384 50
peg-insert-side-v2 3909.180278988091 183731.47311244026 47
pick-place-wall-v2 4162.3110434573155 203953.24112940847 49
pick-out-of-hole-v2 3339.442903196961 166972.14515984806 50
reach-v2 4835.3356478567885 241766.7823928394 50
push-back-v2 382.10794520709084 19105.39726035454 50
push-v2 3558.023228518536 177901.1614259268 50
pick-place-v2 4272.123546496604 213606.1773248302 50
plate-slide-v2 4420.695680887148 221034.7840443574 50
plate-slide-side-v2 4034.975560214308 201748.7780107154 50
plate-slide-back-v2 1171.0356284218753 58551.781421093765 50
plate-slide-back-side-v2 1024.5694094900723 51228.470474503614 50
peg-unplug-side-v2 1229.9545239303345 61497.72619651673 50
soccer-v2 2798.3474815421946 120328.94170631438 43
stick-push-v2 1286.714150871722 64335.7075435861 50
stick-pull-v2 513.1297611247296 24630.22853398702 48
push-wall-v2 4156.398407806063 207819.92039030316 50
reach-wall-v2 4800.020615934053 240001.03079670266 50
shelf-place-v2 2833.0132176530897 141650.66088265448 50
sweep-into-v2 2489.10602743353 112009.77123450885 45
sweep-v2 4350.4701424681125 217523.5071234056 50
window-open-v2 2129.840301628518 106492.0150814259 50
window-close-v2 3121.984379533082 156099.2189766541 50"""

v1 = """assembly-v2 2695649.8468503873 134782492.34251937 50
basketball-v2 39313.77031206153 1965688.5156030767 50
bin-picking-v2 854410.0949367455 42720504.74683727 50
box-close-v2 -8.08363446791972 -388.01445446014657 48
button-press-topdown-v2 896603.3301725581 44830166.50862791 50
button-press-topdown-wall-v2 34654.30097832585 1732715.0489162926 50
button-press-v2 -61.12740761922143 -3056.3703809610715 50
button-press-wall-v2 395866.43165648426 19397455.151167728 49
coffee-button-v2 -60.548973728438085 -3027.4486864219043 50
coffee-pull-v2 889007.1071586343 44450355.35793172 50
coffee-push-v2 163833.84497951748 8191692.248975874 50
dial-turn-v2 -48.279946162612376 -2413.997308130619 50
disassemble-v2 13891.207528852232 694560.3764426116 50
door-close-v2 1876.1579286254907 93807.89643127454 50
door-lock-v2 65065.90793307598 3253295.396653799 50
door-open-v2 18714.682853654605 935734.1426827303 50
door-unlock-v2 -62.50171098557794 -3125.085549278897 50
hand-insert-v2 843179.2720130624 42158963.60065312 50
drawer-close-v2 126.67468666471977 6333.734333235989 50
drawer-open-v2 909.0682114022461 45453.410570112304 50
faucet-open-v2 234575.05247303846 11728752.623651924 50
faucet-close-v2 238640.95108399328 11932047.554199664 50
hammer-v2 195906.22459721135 9795311.229860567 50
handle-press-side-v2 300978.87250369886 15048943.625184942 50
handle-press-v2 348365.353399533 17418267.66997665 50
handle-pull-side-v2 625169.669944944 31258483.4972472 50
handle-pull-v2 426637.3483238925 21331867.416194625 50
lever-pull-v2 133057.36609092035 6652868.304546017 50
peg-insert-side-v2  no successes for this env
pick-place-wall-v2 462800.59681661485 22677229.24401413 49
pick-out-of-hole-v2 543691.5525289807 27184577.626449034 50
reach-v2 562053.5460711892 28102677.30355946 50
push-back-v2 713385.294727087 35669264.73635435 50
push-v2 655525.6007212544 32776280.036062717 50
pick-place-v2 467920.668329615 23396033.41648075 50
plate-slide-v2 855778.55071222 42788927.535611 50
plate-slide-side-v2 -31.66050331288854 -1583.025165644427 50
plate-slide-back-v2 -43.542233550480184 -2177.1116775240093 50
plate-slide-back-side-v2 70839.11542630578 3541955.771315289 50
peg-unplug-side-v2 113520.93329449503 5676046.664724751 50
soccer-v2 525865.3739226413 22612211.078673575 43
stick-push-v2 1567.0385424750339 78351.9271237517 50
stick-pull-v2 3776.3041622504466 181262.59978802144 48
push-wall-v2 686300.8988622428 34315044.94311214 50
reach-wall-v2 588271.3461126653 29413567.305633266 50
shelf-place-v2 477388.16878640035 23869408.439320017 50
sweep-into-v2 636069.0637871345 28623107.870421052 45
sweep-v2 24465.24529702479 1223262.2648512395 50
window-open-v2 25784.676568253442 1289233.828412672 50
window-close-v2 53753.62166979801 2687681.0834899005 50"""

import numpy as np
import matplotlib.pyplot as plt

def plot_scatter_log_scale(list1, list2, title):
    # Assuming list1 and list2 are of the same length, if not, we need to handle it differently
    if len(list1) != len(list2):
        min_length = min(len(list1), len(list2))
        list1 = list1[:min_length]
        list2 = list2[:min_length]

    # Converting string numbers to floats
    list1_floats = [float(num) for num in list1]
    list2_floats = [float(num) for num in list2]

    # Using log scale due to the wide range of values in list1
    plt.figure(figsize=(10, 6))
    plt.scatter(np.log(list1_floats), list2_floats)
    plt.title(title)
    plt.xlabel('Log of Reward')
    plt.ylabel('Env Num')
    plt.grid(True)
    plt.show()


v1 = np.asarray([float(val) for val in extract_numbers(v1)])
v2 = np.asarray([float(val) for val in extract_numbers(v2)])

import scipy.stats as stats

def plot_dist(data):
    mean, std = np.mean(data), np.std(data)

    # Step 2: Create a range of values for plotting the normal distribution
    xmin, xmax = min(data), max(data)
    x = np.linspace(xmin, xmax, 50)

    # Step 3: Create the normal distribution with estimated parameters
    p = stats.norm.pdf(x, mean, std)

    # Step 4: Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=10, density=True, alpha=0.9, color='g')
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mean = {:.2f},  std = {:.2f}".format(mean, std)
    plt.title(title)
    plt.show()

plot_scatter_log_scale(v1, [i for i in range(len(v1))], 'V1')
plot_scatter_log_scale(v2, [i for i in range(len(v2))], 'V2')
print(len(v1))
print(len(v2))'''

