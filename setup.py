import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CATCHM", 
    version="0.1",
    author="Rafael Van Belle",
    author_email="rafael@gmail.com",
    description="A novel network-based credit card fraud detection approach using noderepresentation learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rafaelvanbelle/CATCHM",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'networkx',
        'nodevectors',
        'xgboost',
        'fucc',
        'scikit-learn'
      ],
    classifiers=[
        "Programming Language :: Python :: 3", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)