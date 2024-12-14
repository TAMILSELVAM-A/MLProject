from setuptools import find_packages,setup
from typing import List

HYPEN_DOT_E = "-e ."

def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requriements
    '''
    requirements = []
    with open("requirements.txt") as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_DOT_E in requirements:
            requirements.remove(HYPEN_DOT_E)
    return requirements

setup(
name = "MLPROJECT",
version = "0.0.1",
author = "Tamil",
author_email = "tamilselvamcr72001@gmail.com",
packages = find_packages(),
install_requires = get_requirements('requirements.txt')
)