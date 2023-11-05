# -*- coding: utf-8 -*-
"""
Created on Wed May  4 18:00:28 2022

@author: Gavin
"""

import os

import pandas as pd

# import env.cloud_allocation.lib.simulator.code.simulator.config as config 
import env.simulator.code.simulator.config as config 


from typing import Dict, Union, List

from pandas import DataFrame

DATASET_DIR_LOOKUP = { 'auvergrid' : config.AUVERGRID_DIR,
                       'bitbrains' : config.BITBRAINS_DIR }

FULL_CONTAINER_COLS = ['cpu', 'memory', 'timestamp']
CONTAINER_COLS = ['cpu', 'memory']
OS_COLS = ['os-id']
PM_TYPE_COLS = ['pm-type-id']
VM_TYPE_COLS = ['vm-type-id']

COLUMN_NAMES_LOOKUP = { 'container-data' : FULL_CONTAINER_COLS,
                        'container' : CONTAINER_COLS,
                        'os' : OS_COLS,
                        'pmType' : PM_TYPE_COLS,
                        'vmType' : VM_TYPE_COLS }

def load_init_env_data(test_num: int) -> Dict[str, DataFrame]:
    dataset = config.RUNNING_PARAMS['dataset']
    os_dataset = config.RUNNING_PARAMS['OS-dataset']
    
    env_data_dir = DATASET_DIR_LOOKUP[dataset] + '/' + os_dataset + '/InitEnv/testCase' + str(test_num)
    
    init_env_data = {}
    
    fnames = os.listdir(env_data_dir)
    fnames.sort()
    
    for fname in fnames:
        key = fname[:-4]
        value = __read_init_env_data_file(fname, env_data_dir + '/' + fname)
        
        init_env_data[key] = value   
        
    return init_env_data

def load_valid_init_env_data() -> Dict[str, DataFrame]:
    dataset = config.RUNNING_PARAMS['dataset']
    os_dataset = config.RUNNING_PARAMS['OS-dataset']
    
    env_data_dir = DATASET_DIR_LOOKUP[dataset] + '/' + os_dataset + '/InitEnv/testCase' + '-1'
    
    init_env_data = {}
    
    fnames = os.listdir(env_data_dir)
    fnames.sort()
    
    for fname in fnames:
        key = fname[:-4]
        value = __read_init_env_data_file(fname, env_data_dir + '/' + fname)
        
        init_env_data[key] = value   
        
    return init_env_data



def load_container_data(test_num: int) -> DataFrame:
    dataset = config.RUNNING_PARAMS['dataset']
    os_dataset = config.RUNNING_PARAMS['OS-dataset']
    
    container_data_dir = DATASET_DIR_LOOKUP[dataset] + '/' + os_dataset + '/containerData/testCase{:}.csv'.format(test_num)

    containers = pd.read_csv(container_data_dir, header=None, names=COLUMN_NAMES_LOOKUP['container-data'])
    
    return containers

def load_test_container_data(test_num: int) -> DataFrame:
    dataset = config.RUNNING_PARAMS['dataset']
    os_dataset = config.RUNNING_PARAMS['OS-dataset']
    
    container_data_dir = DATASET_DIR_LOOKUP[dataset] + '/' + os_dataset + '/Test/containerData/testCase{:}.csv'.format(test_num)

    containers = pd.read_csv(container_data_dir, header=None, names=COLUMN_NAMES_LOOKUP['container-data'])
    
    return containers



def load_os_data(test_num: int) -> DataFrame:
    dataset = config.RUNNING_PARAMS['dataset']
    os_dataset = config.RUNNING_PARAMS['OS-dataset']
    
    container_data_dir = DATASET_DIR_LOOKUP[dataset] + '/' + os_dataset + '/OSData/testCase{:}.csv'.format(test_num)

    return pd.read_csv(container_data_dir, header=None, names=COLUMN_NAMES_LOOKUP['os'])

def load_test_os_data(test_num: int) -> DataFrame:
    dataset = config.RUNNING_PARAMS['dataset']
    os_dataset = config.RUNNING_PARAMS['OS-dataset']
    
    container_data_dir = DATASET_DIR_LOOKUP[dataset] + '/' + os_dataset + '/Test/OSData/testCase{:}.csv'.format(test_num)

    return pd.read_csv(container_data_dir, header=None, names=COLUMN_NAMES_LOOKUP['os'])




def __read_init_env_data_file(fname: str, fdir: str) -> Union[List[List[int]], DataFrame]:
    fname = fname[:-4]
    
    if fname == 'pm' or fname == 'vm': 
        
        ids = []
        
        with open(fdir, 'r') as file:
            
            for line in file:
                
                line_ids = [int(val) for val in line.split(',')]
                ids.append(line_ids)
                
        return ids
    
    file_data = pd.read_csv(fdir, header=None, names=COLUMN_NAMES_LOOKUP[fname])
    
    return file_data