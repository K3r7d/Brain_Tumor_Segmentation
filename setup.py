from setuptools import setup, find_packages

setup(
    name="mlproject_forfun",
    version="0.1",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)
# The setup.py file is used to package the project. The find_packages function is used to find all the packages in the src directory.
# The package_dir argument is used to specify that the packages are in the src directory.
# The name of the package is mlproject_forfun and the version is 0.1.