import metaworld
import random

print(metaworld.ML1.ENV_NAMES)  # Check out the available environments

game = "plate-slide-v2"
# game = "plate-slide-side-v2"
# game = "plate-slide-back-v2"
ml1 = metaworld.ML1(game) # Construct the benchmark, sampling tasks

env = ml1.train_classes[game]()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

obs = env.reset()  # Reset environment
a = env.action_space.sample()  # Sample an action
for i in range(1000):
    a = env.action_space.sample()
    obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
    env.render()
input("Press enter to exit")

