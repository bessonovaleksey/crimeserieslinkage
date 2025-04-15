from setuptools import setup, find_packages

setup(
    name='crimeserieslinkage',
    version='2.0',
    packages=find_packages(),
    url='https://github.com/bessonovaleksey/crimeserieslinkage.git',
    license='MIT',
    author='Aleksey A. Bessonov',
    author_email='bestallv@mail.ru',
    description='Statistical methods for identifying serial crimes and related offenders',
    install_requires=[
        'numpy>=1.22.1',
        'pandas>=1.5.1',
        'math',
        'tqdm',
        'scipy>=1.7.3',
        'sklearn>=1.3.0',
        'itertools',
        'igraph>=0.10.2',
        'matplotlib>=3.5.1',
        'datetime'
    ],
)
