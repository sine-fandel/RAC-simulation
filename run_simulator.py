# -*- coding: utf-8 -*-
"""
created on 5 November, 2023

@author: Zhengxin Fang
"""

from time import time

from env.simulator.code.simulator import SimulatorState, Simulator

from env.simulator.code.simulator.io.loading import load_init_env_data, load_container_data, load_os_data, \
                                    load_test_container_data, load_test_os_data, load_valid_init_env_data

from numpy import *

TEST_SETS = 200

def train():
    
    start_time = time()

    test_energy_list = []       # energy consumption of each testCase
    
    for i in range(TEST_SETS):
        print(i)
        init_containers, init_os, init_pms, init_pm_types, init_vms, init_vm_types = load_init_env_data(i).values()

        sim_state = SimulatorState(init_pms, init_vms, init_containers, init_os, init_pm_types, init_vm_types)
        sim = Simulator(sim_state)
        
        input_containers = load_container_data(i)
        input_os = load_os_data(i)
        
        print('Executing testCase {:}...'.format(i))
        
        start = time()
        
        input_containers_tuples = input_containers.apply(lambda x: tuple(x), axis=1).values.tolist()
        input_os_ints = input_os.apply(lambda x: int(x), axis=1).values.tolist()
        for j in range(len(input_containers_tuples)) :
            sim.heuristic_method(input_containers_tuples[j], input_os_ints[j])
            # pm_stats = sim.state.pm_actual_usage[0]
            # print(sim.selected_vm)
            # print(sim.state.vm_pm_mapping[sim.selected_vm])
            # if len(sim.state.pm_actual_usage) >= 541:
            #     # print(sim.selected_vm)
            #     # print(sim.state.vm_pm_mapping[sim.selected_vm])
            #     print(sim.state.pm_actual_usage[540])
        
        # for k in range(len(sim.get_state().pm_actual_usage)):
        #     if sim.get_state().pm_actual_usage[k][0] < 0 or sim.get_state().pm_actual_usage[k][1] < 0 :
        #         print(len(init_pms))
        #         print("bug")

        # for k in range(len(sim.get_state().pm_resources)):
        #     if sim.get_state().pm_resources[k][0] < 0 or sim.get_state().pm_resources[k][1] < 0 :
        #         print(len(init_pms))
        #         print("bug")

        # for k in range(len(sim.get_state().vm_resources)):
        #     if sim.get_state().vm_resources[k][0] < 0 or sim.get_state().vm_resources[k][1] < 0 :
        #         print(len(init_pms))
        #         print("bug")

        test_energy_list.append(sim.get_energy_of_testcase())
        print('Total Energy Consumption: {:}'.format(sim.get_energy_of_testcase()))
        
        end = time()
        
        exec_time = end - start
        
        print('Ran Simulator in {:} Seconds\n'.format(exec_time))
        
    end_time = time()
    
    print('Simulated All testCases in {:} Seconds\n'.format(end_time - start_time))
    print("Average Energy Consumption = {:}".format(mean(test_energy_list)))

def validation():
    
    start_time = time()

    test_energy_list = []       # energy consumption of each testCase
    

    init_containers, init_os, init_pms, init_pm_types, init_vms, init_vm_types = load_init_env_data(0).values()
    sim_state = SimulatorState(init_pms, init_vms, init_containers, init_os, init_pm_types, init_vm_types)
    
    sim = Simulator(sim_state)

    # # validate the warm up
    # init_state = sim.get_state()

    # print(init_state.pm_resources[0])                  # pm_resources are the avaiable resource of pm
    # print(init_state.pm_actual_usage[0])                  # pm_actual_usage are the remaining resources of pm
    # print(init_state.vm_resources[0])                  # vm_resources are the remaining resources of vm
    # print(init_state.current_energy_unit_time)      # current energy consumption
    
    # # test the features

    input_containers = load_container_data(0)
    input_os = load_os_data(0)
    input_containers_tuples = input_containers.apply(lambda x: tuple(x), axis=1).values.tolist()
    input_os_ints = input_os.apply(lambda x: int(x), axis=1).values.tolist()
    for i in range(len(input_containers_tuples)):
        sim.heuristic_method(input_containers_tuples[i], input_os_ints[i])
        for feature in sim.feature_terminal_map:
            print("feature = {:}, value = {:}".format(feature, sim.feature_terminal_map[feature]()))
        
        print("################")

    
        
# validation()

train()