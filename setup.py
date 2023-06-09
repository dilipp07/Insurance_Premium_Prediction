from setuptools import setup,find_packages
from typing import List
def get_requirements(file_path:str)->List[str]:
    '''
    This  function will return list of requirement
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")
        
    return requirements


setup(name="ML-Project",
version='0.0.1',author='Dilip',author_email='dilipkupanda23@gmail.com',
packages=find_packages(),
install_requirements=get_requirements('requirements.txt')  )