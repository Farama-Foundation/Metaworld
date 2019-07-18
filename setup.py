from setuptools import find_packages, setup


# Required dependencies
required = [
    # Please keep alphabetized
    'gym',
    'mujoco-py',
    'numpy-stl',
    'opencv-python',
    'pyquaternion',
]


# Development dependencies
extras = dict()
extras['dev'] = [
    # Please keep alphabetized
    'ipdb',
    'pylint',
    'pytest',
]


setup(
    name='metaworld',
    packages=find_packages(),
    install_requires=required,
    extras_require=extras,
)
