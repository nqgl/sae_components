from pathlib import Path

from setuptools import find_namespace_packages, setup

requirements = Path("requirements.txt").read_text("utf-8").splitlines()

setup(
    name="saeco",
    version="0.0.1",
    description="SAE modular components.",
    long_description=Path("README.md").read_text("utf-8"),
    author="Glen M. Taggart",
    author_email="glenmtaggart@gmail.com",
    url="https://github.com/nqgl/sae-components",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={},
    classifiers=[
        "Programming Language :: Python :: 3.13",
    ],
)
