from os import path
from setuptools import setup, find_packages
from codecs import open

with open(
    path.join(path.abspath(path.dirname(__file__)), "README.md"), encoding="utf-8"
) as f:
    long_description = f.read()

setup(
    name="numpy_int_ops",
    version="0.1.0",
    license="MIT",
    author="Tilmann E. Bartsch",
    url="https://github.com/tebartsch/numpy-int-ops",
    description="Fast integer operations for numpy.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
    ],
    keywords="numpy oneDNN matmul",
    packages=find_packages(include=["numpy_int_ops"]),
    package_data={'': ['libIntOps.so']},
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "cffi",
    ],
)