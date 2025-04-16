from setuptools import setup, find_packages

setup(
    name='crimeserieslinkage',
    version='2.1.0',
    packages=find_packages(),
    url='https://github.com/bessonovaleksey/crimeserieslinkage.git',
    license='Apache Software License',
    author='Aleksey A. Bessonov',
    author_email='bestallv@mail.ru',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    description='Statistical methods for identifying serial crimes and related offenders',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'tqdm',
        'scipy',
        'sklearn',
        'igraph',
        'math',
        'matplotlib',
        'datetime'
    ],
)
