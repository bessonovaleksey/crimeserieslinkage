from setuptools import setup, find_packages

setup(
    name='crimeserieslinkage',
    version='2.1.0',
    packages=find_packages(),
    url='https://github.com/bessonovaleksey/crimeserieslinkage.git',
    license='Apache Software License',
    author='Aleksey A. Bessonov',
    author_email='bestallv@mail.ru',
    description='Statistical methods for identifying serial crimes and related offenders',
    install_requires=[
        'numpy',
        'pandas',
        'tqdm',
        'scipy',
        'sklearn',
        'igraph',
        'matplotlib',
        'datetime'
    ],
)
