# -*- coding: utf-8 -*-
"""
@author: Zhengxin Fang
"""

from copy import deepcopy

from typing import List, Iterable, Union

from pandas import DataFrame

from numpy import array
import numpy as np

from env.simulator.code.simulator.metrics.vm import CPUJustFitVMAllocation
from env.simulator.code.simulator.metrics.pm import FirstFitPMAllocation, BestFitCPUPMAllocation

from env.simulator.code.simulator.config import (AMAZON_PM_TYPES,
                                            AMAZON_VM_TYPES,
                                            VM_CPU_OVERHEAD_RATE,
                                            VM_MEMORY_OVERHEAD)

from enum import Enum

class FeatureEnum(Enum):
    # VM selection features
    CONTAINER_CPU = "CC"
    CONTAINER_MEMORY = "CM"
    LEFT_VM_CPU = "LVC"
    LEFT_VM_MEMORY = "LVM"
    VM_CPU_OVERHEAD = "VCO"
    VM_MEMORY_OVERHEAD = "VMO"

    # PM selection features
    VM_CPU = "VC"
    VM_MEMORY = "VM"
    LEFT_PM_CPU = "LPC"
    LEFT_PM_MEMORY = "LPM"
    PM_CPU = "PC"
    PM_MEMORY = "PM"
    PM_CORE = "PCO"

class SimulatorState():
    '''
    initialize the simulation based on initEnv
    each test correspond to a testCase in initEnv
    '''
    def __init__(self,
                 init_pms: List[List[int]],
                 init_vms: List[List[int]],
                 init_containers: DataFrame,
                 init_os: DataFrame,
                 init_pm_types: DataFrame,
                 init_vm_types: DataFrame,
                 ) -> None:
        
        self.pms = init_pms
        self.vms = init_vms
        self.containers = init_containers
        self.os = init_os
        self.pm_types = init_pm_types
        self.vm_types = init_vm_types
    
        self.max_cpus = []
        self.pm_resources = []
        self.pm_actual_usage = []
        self.vm_resources = []
        
        self.vm_pm_mapping = {}
        self.vm_index_type_mapping ={}
        
        vm_types = self.vm_types
    
        for i, vm_stats in enumerate(self.vms):
            '''
            i: the index of VM type in vmType file
            '''
            vm_type_index = vm_types.loc[i][0]
        
            selected_vm_type = AMAZON_VM_TYPES.loc[vm_type_index]
            
            self.vm_resources.append(array([selected_vm_type[0] * (1 - VM_CPU_OVERHEAD_RATE),
                                            selected_vm_type[1] - VM_MEMORY_OVERHEAD,
                                            None, selected_vm_type[2], vm_type_index]))
            
        for i, (pm_types_row, pm_vms_data) in enumerate(zip(self.pm_types.values, self.pms)):
            pm_type_id = pm_types_row[0]
            
            pm_type_stats = AMAZON_PM_TYPES.loc[pm_type_id]
            
            pm_cpu = pm_type_stats['cpu-max']
            pm_mem = pm_type_stats['memory-max']
            pm_idle_power = pm_type_stats['idle-power']
            pm_max_power = pm_type_stats['max-power']
            pm_core_num = pm_type_stats['cores-num']
            
            ordered_stats = [pm_cpu, pm_mem, pm_idle_power, pm_max_power, pm_core_num, pm_cpu, pm_mem]
            
            self.max_cpus.append(pm_cpu)
            self.pm_resources.append(ordered_stats)
            self.pm_actual_usage.append(ordered_stats)
            
            for vm_index in pm_vms_data:
                vm_type_id = self.vm_types.loc[vm_index]['vm-type-id']
                '''
                os = self.os.loc[vm_index]
                
                self.vm_resources.append([AMAZON_VM_TYPES.loc[vm_type_id]['cpu-max'] * (1 - VM_CPU_OVERHEAD_RATE),
                                        AMAZON_VM_TYPES.loc[vm_type_id]['memory-max'] - VM_MEMORY_OVERHEAD,
                                        os['os-id'], AMAZON_VM_TYPES.loc[vm_type_id]['cores-num']])
                '''
                containers_indices = self.vms[vm_index]
                
                #pm_index = len(self.pm_resources) - 1
                pm_index = i
                
                self.pm_resources[pm_index] = array([self.pm_resources[pm_index][0] - AMAZON_VM_TYPES.loc[vm_type_id][0],
                                            self.pm_resources[pm_index][1] - AMAZON_VM_TYPES.loc[vm_type_id][1],
                                            self.pm_resources[pm_index][2], self.pm_resources[pm_index][3],
                                            self.pm_resources[pm_index][4], self.pm_resources[pm_index][5],
                                            self.pm_resources[pm_index][6]])


                self.pm_actual_usage[pm_index] = array([self.pm_actual_usage[pm_index][0] - (AMAZON_VM_TYPES.loc[vm_type_id]['cpu-max'] * VM_CPU_OVERHEAD_RATE),
                                                        self.pm_actual_usage[pm_index][1] - VM_CPU_OVERHEAD_RATE, self.pm_actual_usage[pm_index][2],
                                                        self.pm_actual_usage[pm_index][3], self.pm_actual_usage[pm_index][4],
                                                        self.pm_actual_usage[pm_index][5], self.pm_actual_usage[pm_index][6]])
                
                self.vm_pm_mapping[vm_index] = pm_index
                self.vm_index_type_mapping[vm_index] = vm_type_id
                
                for container_index in containers_indices:
                    container_stats = self.containers.loc[container_index]
                    
                    #vm_index = len(self.vm_resources) - 1
                    
                    
                    os = self.os.loc[vm_index]
                    self.vm_resources[vm_index] = array([self.vm_resources[vm_index][0] - container_stats['cpu'],
                                                        self.vm_resources[vm_index][1] - container_stats['memory'],
                                                        os['os-id'], self.vm_resources[vm_index][3], self.vm_resources[vm_index][4]])

                    pm_stats = self.pm_actual_usage[pm_index]
                    
                    self.pm_actual_usage[pm_index] = array([pm_stats[0] - container_stats['cpu'],
                                                            pm_stats[1] - container_stats['memory'],
                                                            pm_stats[2], pm_stats[3], pm_stats[4],
                                                            pm_stats[5], pm_stats[6]])

        unit_power_consumption = 0
        
        for pm_resources_bundle, pm_actual_usage_bundle in zip(self.pm_resources, self.pm_actual_usage):
            idle_power = pm_resources_bundle[2]
            max_power = pm_resources_bundle[3]
            
            unit_power_consumption += idle_power
            
            utilisation = 1 - (pm_actual_usage_bundle[0] / pm_resources_bundle[5])
                
            unit_power_consumption += (max_power - idle_power) * (2 * utilisation - (utilisation ** 1.4))


        self.current_energy_unit_time = unit_power_consumption


class Simulator():
    
    def __init__(self, init_state: SimulatorState, test=False) -> None:

        self.initial_test = test
        import copy
        self.state = copy.deepcopy(init_state)      # deep copy: change the self.state would not change init_state 
        if test == True:
            self.state.vm_resources = []
            self.state.max_cpus = []
            self.state.pm_resources = []
            self.state.pm_actual_usage = []
            self.state.vm_resources = []
            
            self.state.vm_pm_mapping = {}
            self.state.vm_index_type_mapping ={}

            self.running_energy_unit_time = 0
        
        else :
            self.running_energy_unit_time = self.state.current_energy_unit_time
            
        self.running_energy_consumption = 0
        self.running_communication_overhead = 0

        self.selected_vm = None                     # the selected VM id
        self.selected_pm = None                     # the selected PM id
        self.container_stats = None                 # the stats of container to be deployed
        
        self.current_timestamp = 0
        
        self.max_cpu = 41600
        self.max_memory = 256000
        
        self.to_allocate_vm_data = None

        self.energy_list = []

    def reset(self, state: SimulatorState) -> None:
        """reset for rotation
        """
        self.state = state
        
        self.running_energy_unit_time = self.state.current_energy_unit_time
        self.running_energy_consumption = 0

        self.selected_vm = None                     # the selected VM id
        self.selected_pm = None                     # the selected PM id
        self.container_stats = None                 # the stats of container to be deployed
        
        self.current_timestamp = 0
        
        self.max_cpu = 41600
        self.max_memory = 256000
        
        self.to_allocate_vm_data = None

        self.energy_list = []   
    
    def heuristic_method(self, container_stats: tuple, container_os: int) -> None:
        '''
        use heuristic method to generate actions
        '''
        self.container_stats = container_stats

        cpu_just_fit_vm = CPUJustFitVMAllocation ()
        ff_pm = FirstFitPMAllocation()

        # VM selection
        vm_selection: dict = cpu_just_fit_vm(self.state, container_stats, container_os)
        self.step_first_layer(vm_selection, *container_stats, container_os)

        # PM selection
        if self.to_allocate_vm_data != None:
            pm_selection: dict = ff_pm(self.state, self.to_allocate_vm_data[1])
            self.step_second_layer(pm_selection, *container_stats, container_os)
    
    def vm_selection(self, func, container_stats, container_os: int) -> list:
        '''vm selection for container
        '''
        container = container_stats.iloc        # container info
        vm_cpu_overhead_rate = VM_CPU_OVERHEAD_RATE
        vm_memory_overhead = VM_MEMORY_OVERHEAD
        # VM features
        vm_index_type_mapping = self.state.vm_index_type_mapping
        amazon_vm_types = np.array(AMAZON_VM_TYPES)[:, 0 : 2]
        if self.initial_test == True:
            """
            TEST
            """
            candidate_vms = amazon_vm_types
        else:
            running_machine = np.array(self.state.vm_resources)[:, 0 : 2]
            candidate_vms = np.append(running_machine, amazon_vm_types, axis=0)
            total_vms = len(running_machine)
        
        vm_mapping = np.where((candidate_vms[:, 0] >= container[0])&(candidate_vms[:, 1] >= container[1]))[0]
        
        avaiable_vms = candidate_vms[(candidate_vms[:, 0] >= container[0])&(candidate_vms[:, 1] >= container[1]), :]
        num_vms = len(avaiable_vms)
        remaining_cpu_capacities = avaiable_vms[:, 0]
        remaining_memory_capacities = avaiable_vms[:, 1]
        
        vm_cpu_overheads = []
    
        for index in vm_mapping:
            if self.initial_test == True:
                """
                TEST
                """
                vm_cpu_overheads.append(amazon_vm_types[index, 0] * vm_cpu_overhead_rate)
            else:
                if index < total_vms:
                    vm_cpu_overheads.append(amazon_vm_types[vm_index_type_mapping[index], 0] * vm_cpu_overhead_rate)
                else:
                    vm_cpu_overheads.append(amazon_vm_types[index - total_vms, 0] * vm_cpu_overhead_rate)

        vm_memory_overheads = np.full(num_vms, vm_memory_overhead)

        # container features
        container_cpus = np.full(num_vms, container[0])
        container_memories = np.full(num_vms, container[1])
        
        merged_features = np.array((container_cpus, container_memories,
                                    remaining_cpu_capacities, remaining_memory_capacities,
                                    vm_cpu_overheads, vm_memory_overheads))
        
        key = np.argmax(func(*merged_features))
        action = { "vm_num": vm_mapping[key] }

        return action

    def pm_selection(self, func) -> list:
        '''pm selection for new created vm
        '''
        vm_cpu_capacity, vm_memory_capacity, vm_used_cpu, vm_used_memory, vm_core = self.to_allocate_vm_data[1]
        largest_priority = float("inf")
        selected_id = -1
        running_machine = np.array(self.state.pm_resources)
        amazon_pm_types = np.array(AMAZON_PM_TYPES)
        total_pm_types = amazon_pm_types[:, : 2]
        amazon_pm_types = np.append(amazon_pm_types, total_pm_types, axis=1)
        if self.initial_test == True:
            candidate_pms = amazon_pm_types
            self.initial_test = False
        else:
            candidate_pms = np.append(running_machine, amazon_pm_types, axis=0)

        pm_mapping = np.where((candidate_pms[:, 0] >= vm_cpu_capacity)&(candidate_pms[:, 1] >= vm_memory_capacity)&(candidate_pms[:, 4] >= vm_core))[0]

        avaiable_pms = candidate_pms[(candidate_pms[:, 0] >= vm_cpu_capacity)&(candidate_pms[:, 1] >= vm_memory_capacity)&(candidate_pms[:, 4] >= vm_core), :]
        num_pms = len(avaiable_pms)

        vm_cpu_capacities = np.full(num_pms, vm_cpu_capacity).astype(np.float64)
        vm_memory_capacities = np.full(num_pms, vm_memory_capacity).astype(np.float64)
        remaining_cpu_capacities = avaiable_pms[: , 0].astype(np.float64)
        remaining_memory_capacities = avaiable_pms[: , 1].astype(np.float64)
        pm_cpu_capacities = avaiable_pms[: , -2].astype(np.float64)
        pm_memory_capacities = avaiable_pms[: , -1].astype(np.float64)
        pm_cores = avaiable_pms[:, 4].astype(np.float64)
        
        merged_features = np.array((vm_cpu_capacities, vm_memory_capacities,
                                    remaining_cpu_capacities, remaining_memory_capacities,
                                    pm_cpu_capacities, pm_memory_capacities, pm_cores))
        # print(merged_features)
        key = np.argmax(func(*merged_features))
        action = { "pm_num": pm_mapping[key]}

        return action


    def step_first_layer(self, action: dict, *container_stats: Iterable) -> None:
        '''action['vm_num'] is the index of a VM
        '''
        vm_num = action['vm_num']

        allocation_results = self.allocate_container_to_vm(vm_num, *container_stats)    
        
        if allocation_results != None:
            self.to_allocate_vm_data = allocation_results

        
    def step_second_layer(self, action: dict, *container_stats: Iterable) -> None:
        pm_num = action['pm_num']
        
        if self.to_allocate_vm_data != None: 
            self.allocate_vm_to_pm(pm_num, *self.to_allocate_vm_data, *container_stats)
            self.to_allocate_vm_data = None
        
    
    def allocate_container_to_vm(self, vm_num: int, *container_stats: Iterable) -> Union[tuple, None]: #assume that vm_num will never be for a full/incompatible VM
        container_cpu, container_memory, container_timestamp, container_os = container_stats 
        
        if self.current_timestamp == 0: self.current_timestamp = container_timestamp
    
        
        if container_timestamp - self.current_timestamp > 0:
            '''
            the energy consumption of an unit time times the time gap
            '''
            self.__update_total_energy(container_timestamp, self.current_timestamp)
            self.current_timestamp = container_timestamp
        
        current_vm_count = len(self.state.vm_resources)
        #print('allocate_container_to_vm >>> current_vm_count: '+str(current_vm_count))
        
        
        if vm_num < current_vm_count: #allocating to existing VM instance
            pm_index = self.state.vm_pm_mapping[vm_num]

            self.state.vm_resources[vm_num] = array([self.state.vm_resources[vm_num][0] - container_cpu,
                                                    self.state.vm_resources[vm_num][1] - container_memory,
                                                    container_os, self.state.vm_resources[vm_num][3], self.state.vm_resources[vm_num][4]])


            '''save the selected pm and vm id
            '''
            self.selected_vm = vm_num
            self.selected_pm = pm_index     
            
            if container_cpu != 0:
                new_remaining_cpu = self.state.pm_actual_usage[pm_index][0] - container_cpu
                self.__update_current_power(pm_index, self.state.pm_actual_usage, new_remaining_cpu)

            
            self.state.pm_actual_usage[pm_index] = array([self.state.pm_actual_usage[pm_index][0] - container_cpu,
                                                        self.state.pm_actual_usage[pm_index][1] - container_memory,
                                                        self.state.pm_actual_usage[pm_index][2],
                                                        self.state.pm_actual_usage[pm_index][3],
                                                        self.state.pm_actual_usage[pm_index][4],
                                                        self.state.pm_actual_usage[pm_index][5],
                                                        self.state.pm_actual_usage[pm_index][6]])
            

            return None
            
        else: #need to create a new VM for container
            self.selected_vm = current_vm_count
            vm_type_index = vm_num - current_vm_count
            
            amazon_vms_selected_type = AMAZON_VM_TYPES.loc[vm_type_index]
            
            selected_vm_type_cpu = amazon_vms_selected_type['cpu-max']
            selected_vm_type_memory = amazon_vms_selected_type['memory-max']
            selected_vm_type_cores = amazon_vms_selected_type['cores-num']
            
            to_allocate_vm = array([selected_vm_type_cpu - container_cpu - (selected_vm_type_cpu * VM_CPU_OVERHEAD_RATE),
                                    selected_vm_type_memory - container_memory - VM_MEMORY_OVERHEAD, 
                                    container_os, selected_vm_type_cores, vm_type_index])
               
            pm_selection_constraints = array([selected_vm_type_cpu, selected_vm_type_memory,
                                              container_cpu + selected_vm_type_cpu * VM_CPU_OVERHEAD_RATE,
                                              container_memory + VM_MEMORY_OVERHEAD, selected_vm_type_cores])
            
    
            return to_allocate_vm, pm_selection_constraints, vm_type_index
        
    
    
    #only called if new vm was created
    def allocate_vm_to_pm(self, pm_num, to_allocate_vm: array,
                          pm_selection_constraints: array,
                          vm_type_index: int, *container_stats: Iterable
                          ) -> None:
        
        container_cpu, container_memory, container_timestamp, container_os = container_stats 
        
        pm_actual_usage = self.state.pm_actual_usage
        
        current_vm_count = len(self.state.vm_resources)
        current_pm_count = len(self.state.pm_actual_usage)
        
        self.state.vm_resources.append(to_allocate_vm)
        
        
        self.state.vm_index_type_mapping[current_vm_count] = vm_type_index

        amazon_vms_selected_type = AMAZON_VM_TYPES.loc[vm_type_index]
        
        selected_vm_type_cpu = amazon_vms_selected_type['cpu-max']
        selected_vm_type_memory = amazon_vms_selected_type['memory-max']
        selected_vm_type_cores = amazon_vms_selected_type['cores-num']
        
        if pm_num >= current_pm_count:
            
            
            self.selected_pm = current_pm_count
            selected_pm_stats = self.__create_new_pm(selected_vm_type_cpu,
                                                     selected_vm_type_memory, 
                                                     selected_vm_type_cores)
            
            

            self.state.pm_resources.append(array([selected_pm_stats[0] - selected_vm_type_cpu,
                                                selected_pm_stats[1] - selected_vm_type_memory, 
                                                selected_pm_stats[2],
                                                selected_pm_stats[3],
                                                selected_pm_stats[4],
                                                selected_pm_stats[5], 
                                                selected_pm_stats[6]]))
            
            self.state.pm_actual_usage.append(array([selected_pm_stats[0] - container_cpu - selected_vm_type_cpu * VM_CPU_OVERHEAD_RATE,
                                          selected_pm_stats[1] - container_memory - VM_MEMORY_OVERHEAD,
                                          selected_pm_stats[2], selected_pm_stats[3], selected_pm_stats[4],
                                          selected_pm_stats[5], selected_pm_stats[6]]))
            
            
            utilisation_numerator = self.state.pm_actual_usage[current_pm_count][0]
            utilisation_denominator = self.state.pm_resources[current_pm_count][5]
            
            utilisation = 1 - (utilisation_numerator / utilisation_denominator)
            
            self.running_energy_unit_time += self.state.pm_actual_usage[current_pm_count][2]
            self.running_energy_unit_time += (selected_pm_stats[3] - selected_pm_stats[2]) * (2 * utilisation - (utilisation ** 1.4))

            self.state.vm_pm_mapping[current_vm_count] = current_pm_count
        
        else:
        
            self.selected_pm = pm_num

            current_pm_cpu_remaining = self.state.pm_resources[pm_num][0] - selected_vm_type_cpu
            current_pm_memory_remaining = self.state.pm_resources[pm_num][1] - selected_vm_type_memory
        

            if container_cpu != 0:
                new_remaining_cpu = self.state.pm_actual_usage[pm_num][0] - container_cpu - selected_vm_type_cpu * VM_CPU_OVERHEAD_RATE
                self.__update_current_power(pm_num, self.state.pm_actual_usage, new_remaining_cpu)

            self.state.pm_resources[pm_num] = array([current_pm_cpu_remaining,
                                          current_pm_memory_remaining,
                                          self.state.pm_actual_usage[pm_num][2],
                                          self.state.pm_actual_usage[pm_num][3],
                                          self.state.pm_actual_usage[pm_num][4],
                                          self.state.pm_actual_usage[pm_num][5],
                                          self.state.pm_resources[pm_num][6]], dtype=object)

            self.state.pm_actual_usage[pm_num] = array([self.state.pm_actual_usage[pm_num][0] - container_cpu - selected_vm_type_cpu * VM_CPU_OVERHEAD_RATE,
                                                        self.state.pm_actual_usage[pm_num][1] - container_memory - VM_MEMORY_OVERHEAD,
                                                        self.state.pm_actual_usage[pm_num][2], self.state.pm_actual_usage[pm_num][3],
                                                        self.state.pm_actual_usage[pm_num][4], self.state.pm_actual_usage[pm_num][5],
                                                        self.state.pm_actual_usage[pm_num][6],], dtype=object)

            self.state.vm_pm_mapping[current_vm_count] = pm_num
    
    
    def  __create_new_pm(self,
                        vm_cpu: float,
                        vm_memory: int,
                        vm_core: int
                        ) -> int:
        
        selected_type = -1
        
        best_current_utilisation_cpu = 0
        
        required_cpu = vm_cpu
        required_memory = vm_memory

        pm_types_values = AMAZON_PM_TYPES.values

        for i, pm_type in enumerate(pm_types_values):
            
            if required_cpu < pm_type[0] and required_memory < pm_type[1] and pm_type[4] >= vm_core:
                
                current_utilisation_cpu = (pm_type[0] - required_cpu) / pm_type[0]
                
                if best_current_utilisation_cpu < current_utilisation_cpu: 
                    selected_type = i
                    best_current_utilisation_cpu = current_utilisation_cpu

        if selected_type < 0 or selected_type > len(pm_types_values): selected_type = 0
        
        new_pm = array([pm_types_values[selected_type][0], pm_types_values[selected_type][1],
                        pm_types_values[selected_type][2], pm_types_values[selected_type][3],
                        pm_types_values[selected_type][4], pm_types_values[selected_type][0],
                        pm_types_values[selected_type][1]])
        
        return new_pm


		
    def __update_current_power(self,
                               selected_pm: int,
                               actual_usage: list,
                               new_cpu_remaining: float,
                               ) -> bool:
        
        previous_utilisation = (actual_usage[selected_pm][5] - actual_usage[selected_pm][0]) / actual_usage[selected_pm][5]
        new_utilisation = (actual_usage[selected_pm][5] - new_cpu_remaining) / actual_usage[selected_pm][5] 

        max_power = actual_usage[selected_pm][3]
        min_power = actual_usage[selected_pm][2]
        
        power_draw_delta = max_power - min_power
        new_power_draw = power_draw_delta * (2 * new_utilisation - (new_utilisation ** 1.4) - 2 * previous_utilisation + (previous_utilisation ** 1.4))
        self.running_energy_unit_time += new_power_draw

        return True
    
    
    
    def __update_total_energy(self,
                              previous_timestamp: int,
                              current_timestamp: int,
                              ) -> bool:
        
        self.running_energy_consumption += self.running_energy_unit_time * abs(previous_timestamp - current_timestamp) / 3.6e6
        

        return True
        
    def get_energy_of_testcase(self) -> List:
        return self.energy_list[-1]
            
    def get_state(self) -> SimulatorState:
        return self.state
    
    # ###################################################
    # # Terminal Nodes
    # ###################################################
    # def feature_container_cpu(self) -> float:
    #     return self.container_stats[0]
    
    # def feature_container_memory(self) -> float:
    #     return self.container_stats[1]

    # def feature_left_vm_cpu(self) -> float:
    #     if self.selected_vm < len(self.state.vm_resources) - 1:
    #         return self.state.vm_resources[self.selected_vm][0]
    #     else:
    #         amazon_vms_selected_type = AMAZON_VM_TYPES.loc[self.state.vm_resources[self.selected_vm][-1]]
    #         return amazon_vms_selected_type['cpu-max']
    
    # def feature_left_vm_memory(self) -> float:
    #     if self.selected_vm < len(self.state.vm_resources) - 1:
    #         return self.state.vm_resources[self.selected_vm][1]
    #     else:
    #         amazon_vms_selected_type = AMAZON_VM_TYPES.loc[self.state.vm_resources[self.selected_vm][-1]]
    #         return amazon_vms_selected_type['memory-max']

    # def feature_vm_cpu_overhead(self) -> float:
    #     amazon_vms_selected_type = AMAZON_VM_TYPES.loc[self.state.vm_resources[self.selected_vm][-1]]
    #     return amazon_vms_selected_type['memory-max'] * VM_CPU_OVERHEAD_RATE 

    # def feature_vm_memory_overhead(self) -> float:
    #     return VM_MEMORY_OVERHEAD

    # def feature_vm_cpu(self) -> float:
    #     amazon_vms_selected_type = AMAZON_VM_TYPES.loc[self.state.vm_resources[self.selected_vm][-1]]
    #     return amazon_vms_selected_type['cpu-max']
    
    # def feature_vm_memory(self) -> float:
    #     amazon_vms_selected_type = AMAZON_VM_TYPES.loc[self.state.vm_resources[self.selected_vm][-1]]
    #     return amazon_vms_selected_type['memory-max']
    
    # def feature_left_pm_cpu(self) -> float:
    #     return self.state.pm_actual_usage[self.selected_pm][0]
    
    # def feature_left_pm_memory(self) -> float:
    #     return self.state.pm_actual_usage[self.selected_pm][1]
    
    # def feature_pm_cpu(self) -> float:
    #     return self.state.pm_resources[self.selected_pm][-2]
        
    # def feature_pm_memory(self) -> float:
    #     return self.state.pm_resources[self.selected_pm][-1]
    
    # def feature_pm_core(self) -> float:
    #     return self.state.pm_resources[self.selected_pm][-3]
