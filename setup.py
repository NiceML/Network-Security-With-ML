from setuptools import setup, find_packages
from typing import List

def get_requirements() -> List[str]:
    """
    This function returns the list of requirements
    """
    requirement_list: List[str] = []
    try:
        with open('requirements.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and not requirement.startswith('#'):
                    if requirement != '-e .':
                        requirement_list.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")
    
    return requirement_list

setup(
    name="networksecurity",
    version="0.0.1",
    author="NiceML",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=get_requirements(),
)
