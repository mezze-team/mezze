# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2017 JHU/APL
#

from setuptools import setup, find_packages

#
# Install the package using distutils
#

setup(
    name='mezze',
    version='0.0.1',
    description='Mezze Quantum Simulator',
    author='Johns Hopkins University Applied Physics Laboratory',
    author_email='kevin.schultz@jhuapl.edu',
    url='http://jhuapl.edu',
    packages=find_packages(),
    install_requires=[
        'cirq',
        'numpy',
        'scipy',
    ],
    extras_require={
        'zne': ['mitiq','tensorflow','tensorflow-quantum'],
        'tfq': ['tensorflow','tensorflow-quantum'],
        'qsim': ['qsimcirq']
    },
    classifiers=[
        'Development Status :: Beta',
        'Environment :: Console',
        'Operating System :: POSIX',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python'
    ],
)
