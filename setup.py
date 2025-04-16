from setuptools import setup

setup(
    name='crimeserieslinkage',
    version='1.0',
    author='Aleksey A. Bessonov',
    author_email='bestallv@mail.ru',
    description='Statistical methods for identifying serial crimes and related offenders',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/bessonovaleksey/crimeserieslinkage.git',
    packages=['crimeserieslinkage'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'tqdm',
        'scipy',
        'scikit-learn',
        'igraph',
        'matplotlib',
        'datetime'
    ],
)
