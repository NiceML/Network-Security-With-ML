import os
import sys
import dill
import numpy as np
import yaml
from networksecurity.exception import NetworkSecurityException
from networksecurity.logging import logging


def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object to a file using dill
    
    Args:
        file_path: Path where the object will be saved
        obj: Python object to save
    """
    try:
        logging.info("Entered the save_object method of utils")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save_object method of utils")
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_object(file_path: str) -> object:
    """
    Load a Python object from a file using dill
    
    Args:
        file_path: Path to the saved object file
        
    Returns:
        The loaded Python object
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def save_numpy_array_data(file_path: str, array: np.array) -> None:
    """
    Save numpy array data to file
    
    Args:
        file_path: Path where the array will be saved
        array: Numpy array to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_numpy_array_data(file_path: str) -> np.array:
    """
    Load numpy array data from file
    
    Args:
        file_path: Path to the numpy array file
        
    Returns:
        Loaded numpy array
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Write content to a YAML file
    
    Args:
        file_path: Path where the YAML file will be saved
        content: Content to write to the file
        replace: Whether to replace existing file
    """
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def read_yaml_file(file_path: str) -> dict:
    """
    Read content from a YAML file
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Content of the YAML file as dictionary
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
