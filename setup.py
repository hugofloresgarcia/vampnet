from setuptools import find_packages
from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()


setup(
    name="vampnet",
    version="1.0.0dev0",
    classifiers=[],
    description="yet another sound synthesizer.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="hugo flores garcia",
    author_email="hugggofloresgarcia@gmail.com",
    license="MIT",
    packages=find_packages(),
    package_data={"assets": ["assets/*", "assets/*/*"]},
    install_requires=install_requires,
    extras_require={
        "tests": [
            "pytest",
            "pytest-cov",
            "line_profiler",
        ],
    },
)
