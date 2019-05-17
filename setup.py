from setuptools import find_packages, setup


# Required dependencies
required = [
    # Please keep alphabetized
    'gym==0.12.1',
    'mujoco-py==1.50.1.68',
    'pyquaternion==0.9.5',
]


# Development dependencies
extras = dict()
extras['dev'] = [
    # Please keep alphabetized
    'ipdb',
    'pylint',
]


setup(
    name='multiworld',
    packages=find_packages(),
    install_requires=required,
    extras_require=extras,
)
