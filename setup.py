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
    author="Hugo Flores García, Prem Seetharaman",
    author_email="hfgacrcia@descript.com",
    url="https://github.com/hugofloresgarcia/vampnet",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch",
        "argbind>=0.3.2",
        "numpy==1.22",
        # "audiotools @ git+https://github.com/descriptinc/audiotools.git@f35914b5b3c6f1bf589cd09481478d741538828e",
        # "dac @ git+https://github.com/descriptinc/descript-audio-codec.git",
        "gradio", 
        "tensorboardX",
        "loralib"
    ],
)
