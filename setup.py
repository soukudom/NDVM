from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="ndvm",
    version="0.1.1",
    author="Dominik Soukup",
    author_email="",
    url="",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
)
