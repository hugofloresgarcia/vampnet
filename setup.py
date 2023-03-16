from setuptools import find_packages
from setuptools import setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="vampnet",
    version="0.0.1",
    classifiers=[
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Topic :: Artistic Software",
        "Topic :: Multimedia",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Multimedia :: Sound/Audio :: Editors",
        "Topic :: Software Development :: Libraries",
    ],
    description="Generative Music Modeling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Hugo Flores García",
    author_email="hfgacrcia@descript.com",
    url="https://github.com/descriptinc/lyrebird-vampnet",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch<=1.11.0",
        "argbind>=0.3.2",
        "pytorch-ignite",
        "rich",
        "audiotools @ git+https://github.com/descriptinc/lyrebird-audiotools.git@0.6.0",
        "tqdm",
        "tensorboard",
        "google-cloud-logging==2.2.0",
        "torchmetrics>=0.7.3",
        "einops",
    ],
)
