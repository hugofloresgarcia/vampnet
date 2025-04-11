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
        "torch==2.4.1",
        "argbind>=0.3.2",
        "numpy==1.23",
        "wavebeat @ git+https://github.com/hugofloresgarcia/wavebeat",
        "lac @ git+https://github.com/hugofloresgarcia/lac.git",
        "descript-audiotools @ git+https://github.com/hugofloresgarcia/audiotools.git",
        "gradio", 
        "loralib",
        "torch_pitch_shift",
        "plotly",
        "pydantic==2.10.6",
        "spaces",
    ],
)
