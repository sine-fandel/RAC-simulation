# -*- coding: utf-8 -*-
"""
Created on Wed May 11 21:42:36 2022

@author: Gavin
"""

# from env.cloud_allocation.lib.simulator.code.simulator.metrics import Metric

# from env.cloud_allocation.lib.simulator.code.simulator.config import AMAZON_VM_TYPES, VM_CPU_OVERHEAD_RATE, VM_MEMORY_OVERHEAD

from env.simulator.code.simulator.metrics import Metric

from env.simulator.code.simulator.config import AMAZON_VM_TYPES, VM_CPU_OVERHEAD_RATE, VM_MEMORY_OVERHEAD


'''
all vm metrics should have the following input format:
    0 container_cpu
    1 container_memory
    2 vm_cpu_remaining
    3 vm_memory_remaining
    4 UNDECIDED
'''

def container_vm_compatible(container_cpu: float, container_memory: int, container_os: int,
                            vm_cpu_remaining: float, vm_memory_remaining: int, vm_os: int,
                            ) -> bool:

    return vm_cpu_remaining >= container_cpu and vm_memory_remaining >= container_memory and vm_os == container_os 


class CPUJustFitVMAllocation(Metric):
    
    def __call__(self, *inputs: tuple) -> float:
        state, container_stats, container_os = inputs
        
        sim_state = state
        
        selected_vm = -1
        best_candidate_vm_score = 0
        
        vm_resources = sim_state.vm_resources
        pm_actual_usage = sim_state.pm_actual_usage
        
        num_vms = len(vm_resources)
        
        for i, vm_stats in enumerate(vm_resources):

            vm_cpu_remaining, vm_memory_remaining, vm_os, vm_core, vm_type = vm_stats
            pm_stats = pm_actual_usage[sim_state.vm_pm_mapping[i]]

            if container_vm_compatible(*container_stats[ : 2], container_os, *vm_stats[ : 3]):
                # if container_stats[0] > pm_stats[0] or container_stats[1] > pm_stats[1]:
                #     print("bug here")
                #     print(container_stats[0], vm_cpu_remaining, pm_stats[0])
                #     print(container_stats[1], vm_memory_remaining, pm_stats[1])

                vm_score = vm_cpu_remaining - container_stats[0]
                
                if vm_score > best_candidate_vm_score:
                    selected_vm = i
                    best_candidate_vm_score = vm_score
            
            # if selected_vm != -1 and sim_state.vm_pm_mapping[selected_vm] == 0:
            #     print(selected_vm)
            #     pm_stats = pm_actual_usage[sim_state.vm_pm_mapping[selected_vm]]
            #     print(pm_stats)
            
        
        if selected_vm == -1:
            
            for i, vm_stats in enumerate(AMAZON_VM_TYPES.values):
                vm_os = container_os
                
                vm_cpu_remaining, vm_memory_remaining, _ = vm_stats
                
                vm_cpu_remaining *= 1 - VM_CPU_OVERHEAD_RATE
                vm_memory_remaining -= VM_MEMORY_OVERHEAD
                
                wrap = (vm_cpu_remaining, vm_memory_remaining, vm_os)
                
                if container_vm_compatible(*container_stats[ : 2], container_os, *wrap):
                    selected_vm = num_vms + i
                    break

        # pm_stats = pm_actual_usage[0]
        # print(pm_stats)
        
        return { 'vm_num' : selected_vm }



class MemoryJustFitVMAllocation(Metric):
    
    def __call__(self, *inputs: tuple) -> float:
        state, container_stats, container_os = inputs
        
        sim_state = state.state
        
        selected_vm = -1
        best_candidate_vm_score = 0
        
        vm_resources = sim_state.vm_resources
        
        num_vms = len(vm_resources)
        
        for i, vm_stats in enumerate(vm_resources):
            
            vm_cpu_remaining, vm_memory_remaining, vm_os = vm_stats
            
            if container_vm_compatible(*container_stats[:2], container_os, *vm_stats):
                vm_score = vm_memory_remaining - container_stats[1]
                if vm_score > best_candidate_vm_score:
                    selected_vm = i
                    best_candidate_vm_score = vm_score
        
        if selected_vm == -1:
            
            for i, vm_stats in enumerate(AMAZON_VM_TYPES.values):
            
                vm_os = container_os
                
                vm_cpu_remaining, vm_memory_remaining, _ = vm_stats
                
                vm_cpu_remaining *= 1 - VM_CPU_OVERHEAD_RATE
                vm_memory_remaining -= VM_MEMORY_OVERHEAD
                
                wrap = (vm_cpu_remaining, vm_memory_remaining)
                
                if container_vm_compatible(*container_stats[:2], container_os, *wrap, vm_os):
                    selected_vm = num_vms + i
                    break
        
        return { 'vm_num' : selected_vm }