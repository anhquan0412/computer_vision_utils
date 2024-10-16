from setuptools import setup, find_packages


with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()


setup(
    name='cv_utils',
    version='0.1.5',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'tqdm',
        'Pillow',
    ],
    entry_points={
        'console_scripts': [
            'rotate-images-portrait=cv_utils.rotate_images_portrait:main',
            'crop-images-from-file=cv_utils.crop_images:main',
        ],
    },
    author='Quan Tran',
    description='A collection of common utility functions for computer vision tasks.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/anhquan0412/computer_vision_utils',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    license='Apache License 2.0',
    python_requires='>=3.8',
)