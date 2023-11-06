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

from optim.single_tree_gp import SingleTreeGP

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-s", "--SEED", help="seed", dest="seed", type=int, default="0")
args = parser.parse_args()


singlegp = SingleTreeGP(args.seed)

pop, log, hof = singlegp.perturb()

# import pickle

# from env.simulator.code.simulator.metrics.pm import FirstFitPMAllocation


# singlegp = SingleTreeGP(0)
# model = f"/Users/fangzhengxin/Downloads/Ph.D. Project/simulation/RAC/results/model/hof_1699090740.2578912.pkl"
# with open(model, 'rb') as file:
#     hof = pickle.load(file)

# individual = hof[0]

# func = singlegp.toolbox.compile(expr=individual)
# hist_fitness = []       # save all fitness
# print(str(individual))
# test_case_num = 100

# for case in range(test_case_num):
#     init_containers, init_os, init_pms, init_pm_types, init_vms, init_vm_types = load_init_env_data(case).values()
#     # Warm up the simulation
#     sim_state = SimulatorState(init_pms, init_vms, init_containers, init_os, init_pm_types, init_vm_types)
#     sim = Simulator(sim_state)

#     # load the training data
#     input_containers = load_container_data(case)
#     input_os = load_os_data(case)

#     for i in range(len(input_os)) :
#         # sim.heuristic_method(tuple(input_containers.iloc[i].to_list()), input_os.iloc[i]["os-id"])
#         candidates = sim.vm_candidates(tuple(input_containers.iloc[i].to_list()), input_os.iloc[i]["os-id"])
#         largest_priority = float("inf")
#         selected_id = -1
#         for vm in candidates:
#             if largest_priority > func(*tuple(vm[ : -1])):
#                 selected_id = vm[-1]
#                 largest_priority = func(*vm[ : -1])

#         action = {"vm_num": selected_id}
#         sim.step_first_layer(action, *tuple(input_containers.iloc[i].to_list()), input_os.iloc[i]["os-id"])
#         ff_pm = FirstFitPMAllocation()
#         if sim.to_allocate_vm_data != None:
#             pm_selection: dict = ff_pm(sim.state, sim.to_allocate_vm_data[1])
#             sim.step_second_layer(pm_selection, *tuple(input_containers.iloc[i].to_list()), input_os.iloc[i]["os-id"])
    
#     # # # # # # # # # # # # # # # # # #  
#     # debug
#     # # # # # # # # # # # # # # # # # # 
#     for k in range(len(sim.get_state().pm_actual_usage)):
#         if sim.get_state().pm_actual_usage[k][0] < 0 or sim.get_state().pm_actual_usage[k][1] < 0:
#             print("total={:}".format(len(sim.get_state().pm_actual_usage)))
#             print("id={:}".format(k))
#             print("bug")

#     print(sim.running_energy_unit_time)
#     hist_fitness.append(sim.running_energy_unit_time)





