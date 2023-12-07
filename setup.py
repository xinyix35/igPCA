import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="igPCA",
    version="1.0.1",
    author="Xinyi Xie",
    author_email="xinyix35@uw.edu",
    description="Integrative Generalized Principle Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xinyix35/igPCA",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
