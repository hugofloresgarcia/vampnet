from setuptools import find_packages
from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="vampnet",
    version="0.0.1",
    description="generative musical instrument system.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Hugo Flores García, Prem Seetharaman",
    author_email="huferflo@gmail.com",
    url="https://github.com/hugofloresgarcia/vampnet",
    license="MIT",
    packages=find_packages(),
    install_requires=requirements,
)
