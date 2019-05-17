# Metaworld


Metaworld is a multitask manipulation environment that is currently under development. If you’d like to contribute to creating new environments or running algorithms on these environments, please let us know so we can coordinate efforts.


# Basics
Each environment in /metaworld/multiworld/envs/mujoco/sawyer_xyz is an gym environment. To play with the environment, for example, do 


```
 python /metaworld/multiworld/envs/mujoco/sawyer_xyz/button_press_topdown_6dof.py
```

# Testing
The /metaworld/tests/multiworld/envs/mujoco/sawyer_xyz/test_sawyer_parameterized.py file quickly runs every environment with pytest. To run tests, navigate to root dir and run 

```
pytest
```


# Running Demo code for existing environment

/metaworld/scripts/demo_sawyer.py is a command line program to run demo code for environment in /metaworld/multiworld/envs/mujoco/sawyer_xyz

To run this program, you need to know the environment name and use command 

`demo_sawyer.py --env your_env_name`

for example:


```
 ./scripts/demo_sawyer.py --env SawyerNutAssembly6DOFEnv
```


# Pull Request Guidelines


If you would like to contribute *environments*, please submit a pull request. We require the following guidelines for a new environment:


## 1.
Take a look at the spreadsheet to find an environment that doesn’t exist yet. There are task ideas on the spreadsheet.


## 2. Base environment
Your environment should inherit from SawyerXYZEnv in base.py and Multienv in multienv.py. Your class init file should begin like this:


class SawyerExample6DOFEnv(
    SawyerXYZEnv,
    MultitaskEnv,
    Serializable,
    metaclass=abc.ABCMeta,
):
        SawyerXYZEnv.__init__(self, ...)
        MultitaskEnv.__init__(self, ... )




## 3. XML Files
Your environment XML file should include sawyer_xyz_base.xml and share_config.xml


## 4.[a] Action space options
Your environment should implement the following control modes: 3DOF, 4DOF, and 6DOF (quaternion) control. These modes should be controlled by the rotMode class attribute.[b]


## 5. Train a learned policy, and provide a link to the trained policy to the Metaworld spreadsheet.
The task should be learnable with any RL algorithm of your choice.


## 6. (Optional)
Add your name to the Contributions.txt file. The purpose of this file is to track contributors to this project, so in future papers or reports, we can give proper credit to contributors.


## 7. (Optional) Collect a few demonstrations for your task, and add a link to those demonstrations.
This is helpful practice anyway when determining if your new environment is learnable.


## 8. (Optional) We also welcome help contributing to adding textures for the existing environments.
[a]Also say that they need to define a shaped reward function? Or both a shaped reward function and a success rate metric?
[b]Maybe also ask for the contributor to provide a reasonable camera viewpoint for future vision-based experiments?# multiworld
Multitask Environments for RL
