from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='crimeserieslinkage',
    version='2.0',
    packages=find_packages(),
    url='https://github.com/bessonovaleksey/crimeserieslinkage.git',
    license='MIT',
    author='Aleksey A. Bessonov',
    author_email='bestallv@mail.ru',
    description='Statistical methods for identifying serial crimes and related offenders',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache-2.0 License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy>=1.22.1',
        'pandas>=1.5.1',
        'tqdm',
        'scipy>=1.7.3',
        'sklearn>=1.3.0',
        'igraph>=0.10.2',
        'matplotlib>=3.5.1',
        'datetime'
    ],
)
