from setuptools import setup, find_packages

setup(
    name="PyWSI",
    version="0.1.0",
    author="Gregory Verghese",
    author_email="gregory.verghese@gmail.com",
    description="A Python package for preprocessing pathology whole slide images.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gregoryverghese/PySlide",
    packages=find_packages(),
    install_requires=[
        "einops",
        "h5py",
        "huggingface_hub",
        "lmdb",
        "matplotlib",
        "pandas",
        "rocksdb",
        "scipy",
        "seaborn",
        "scikit-image",
        "tensorflow",
        "timm",
        "webdataset"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.12",
)

