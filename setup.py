import os
import platform
import sys

import pkg_resources
from setuptools import find_packages, setup

def read_version(fname="infarctimage.py"):
    with open(fname, encoding="utf-8") as f:
        content = f.readlines()
        for line in content:
            if "__version__" in line:
                return line.split("=")[1].strip().strip("\"")
    return "0.0.1"

requirements = []
setup(
    name="G0-InfarctImage",
    py_modules=["infarctimage"],
    version=read_version(),
    description="InfarctImage is a LoRA-based model fine-tuned on Stable Diffusion 2.1, designed to generate realistic images of people simulating a heart attack. This model was developed as part of a study on synthetic dataset generation for human activity recognition and medical emergency monitoring applications.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.10",
    author="Gabriel Rojas (Gavit0) - G0",
    url="https://github.com/Turing-IA-IHC/InfarctImage",
    license="MIT",
    packages=find_packages(exclude=["notebooks*"]),
    include_package_data=True,
    install_requires=requirements
    + [
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    extras_require={
        "extra": ["kaggle"],
    },
)