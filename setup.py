from setuptools import setup, find_packages


setup(
    name="Machine Vision Toolkit",
    version='0.1.0',
    description="A library for machine vision, artificial intelligence, and deep learning.",
    author="Marco Carletti",
    author_email="marco.carletti.89@gmail.com",
    url="https://github.com/mcarletti/machinevisiontoolkit",
    packages=find_packages(),
    install_requires=[
        "tomli",
        "pyaml",
        "torch",
        "torchinfo",
        "tqdm",
        "opencv-python",
        "packaging",
        "tensorboard",
        "six",
    ],
)
