from setuptools import setup, find_packages

setup(
    name = "RVSNUpy",
    version = "1.0.0",
    author = "Taewan Kim",
    description = "A redshift measurement pacakge based on inverse variance weighted cross-correlation",
    package_dir = {"":"src"},
    packages=find_packages(where="src"),
    python_requires=">=3.0",
    install_requires=["numpy", "scipy", "astropy", "joblib", "tqdm", "pandas", "matplotlib"],
    classifiers=["Programing Language :: Python :: 3",
                 "Operating System :: OS Independent"]
)