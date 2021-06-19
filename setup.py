from setuptools import setup, find_packages

setup(
    name='rlci',
    version='0.0.1',
    description='Reinforcement learning configuration interaction on small FCI Hamiltonians',
    long_description=open('README.md').read(),
    packages=find_packages(),    
    license='MIT',
    python_requires=">=3.4",
    install_requires=['numpy >= 1.20'],
)


