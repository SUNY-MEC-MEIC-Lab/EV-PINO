from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='EVPINO',
    version='0.1.0',
    author='Hansol Lim',
    author_email='hansol.lim@stonybrook.edu',
    description='EV-PINO: Physics-Informed Neural Operator for Electric Vehicle Power Prediction',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/SUNY-MEC-MEIC-Lab/EV-PINO--A-Hybrid-Surrogate-for-EV-Parameter-and-Power-Estimation-Using-PINO',
    packages=find_packages(),
    install_requires=required,
    python_requires='==3.10.16',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    entry_points={
        'console_scripts': [
            'evpino-train=EVPINO_train:main',
            'evpino-test=EVPINO_test:main',
        ],
    },
)
