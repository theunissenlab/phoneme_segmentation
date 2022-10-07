#/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

requirements = [
]

setup(
    name='phoneme_segmentation',
    version='0.1.0',
    description="Code for 2022 Diphone Paper",
    author="Lily Xue Gong",
    author_email='lilyxuegong@berkeley.edu',
    packages=[
        'Diphone',
    ],
    package_dir={'phoneme_segmentation':
                 'phoneme_segmentation'},
    install_requires=requirements,
    license="BSD license",
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)

