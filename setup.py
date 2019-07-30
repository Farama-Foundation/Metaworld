from setuptools import find_packages, setup


# Required dependencies
required = [
    # Please keep alphabetized
    'gym==0.12.1',
    'mujoco-py<2.1,>=2.0',
    'numpy-stl>=2.10.1',
    'opencv-python>=4.1.0.25',
    'pyquaternion==0.9.5',
]


# Development dependencies
extras = dict()
extras['dev'] = [
    # Please keep alphabetized
    'ipdb',
    'pylint',
    'pytest>=3.6',
]


setup(
    name='metaworld',
    packages=find_packages(),
    install_requires=required,
    extras_require=extras,
)
