# -*- coding: utf-8 -*-
"""
Created on Wed May 11 21:43:03 2022

@author: Gavin
"""

# from env.cloud_allocation.lib.simulator.code.simulator.metrics import Metric

# from env.cloud_allocation.lib.simulator.code.simulator.config import (AMAZON_PM_TYPES,
#                                                                       VM_CPU_OVERHEAD_RATE,
#                                                                       VM_MEMORY_OVERHEAD)

from env.simulator.code.simulator.metrics import Metric

from env.simulator.code.simulator.config import (AMAZON_PM_TYPES,
                                                                      VM_CPU_OVERHEAD_RATE,
                                                                      VM_MEMORY_OVERHEAD)

def vm_pm_compatible(pm_cpu_remaining: float, pm_memory_remaining: int, pm_core: int,
                     vm_cpu_capacity: float, vm_memory_capacity: int, vm_core: int
                     ) -> bool:
    return pm_cpu_remaining >= vm_cpu_capacity and pm_memory_remaining >= vm_memory_capacity and pm_core >= vm_core



class FirstFitPMAllocation(Metric):
    
    def __call__(self, *inputs) -> float:
        state, pm_selection_constraints = inputs
        vm_cpu_capacity, vm_memory_capacity, vm_used_cpu, vm_used_memory, vm_core = pm_selection_constraints
        
        pm_resources = state.pm_resources
        num_pms = len(pm_resources)
        
        vm_wrapped = (vm_cpu_capacity, vm_memory_capacity, vm_core)
        
        selected_pm = -1
        
        for pm_count, pm_stats in enumerate(pm_resources):
            pm_cpu_remaining = pm_stats[0]
            pm_memory_remaining = pm_stats[1]
            pm_core = pm_stats[4]
            
            pm_wrapped = (pm_cpu_remaining, pm_memory_remaining, pm_core)
            if vm_pm_compatible(*pm_wrapped, *vm_wrapped):
                # print(pm_stats)
                selected_pm = pm_count
                break
                
        if selected_pm == -1:
            
            for i, pm_stats in enumerate(AMAZON_PM_TYPES.values):
                
                pm_cpu_remaining, pm_memory_remaining, _, _, pm_cores = pm_stats
                
                pm_wrapped = (pm_cpu_remaining, pm_memory_remaining, pm_cores)
                
                if vm_pm_compatible(*pm_wrapped, *vm_wrapped):
                    selected_pm = num_pms + i
                    break

        return { 'pm_num' : selected_pm }
    
    
    
class BestFitCPUPMAllocation(Metric):
    
    def __call__(self, *inputs) -> float:
        state, pm_selection_constraints = inputs
        vm_cpu_capacity, vm_memory_capacity, vm_used_cpu, vm_used_memory, vm_core = pm_selection_constraints
        
        pm_resources = state.pm_resources
        
        num_pms = len(pm_resources)
        
        vm_wrapped = (vm_cpu_capacity, vm_memory_capacity, vm_core)
        
        selected_pm = -1
        min_observed_diff = float('inf')
        
        for pm_count, pm_stats in enumerate(pm_resources):
            pm_cpu_remaining = pm_stats[0]
            pm_memory_remaining = pm_stats[1]
            pm_core = pm_stats[4]
            
            pm_wrapped = (pm_cpu_remaining, pm_memory_remaining, pm_core)
            
            candidate_diff = pm_cpu_remaining - vm_cpu_capacity
            
            is_new_min = candidate_diff < min_observed_diff 
            
            if vm_pm_compatible(*pm_wrapped, *vm_wrapped) and is_new_min:
                selected_pm = pm_count
                min_observed_diff = candidate_diff
                
        if selected_pm == -1:
            
            for i, pm_stats in enumerate(AMAZON_PM_TYPES.values):
                
                pm_cpu_remaining, pm_memory_remaining, _, _, pm_cores = pm_stats
                
                pm_wrapped = (pm_cpu_remaining, pm_memory_remaining, pm_cores)
                
                candidate_diff = pm_cpu_remaining - vm_cpu_capacity
                
                is_new_min = candidate_diff < min_observed_diff 
                
                if vm_pm_compatible(*pm_wrapped, *vm_wrapped) and is_new_min:
                    selected_pm = num_pms + i
                    min_observed_diff = candidate_diff
            
        return { 'pm_num' : selected_pm }
    


class BestFitMemoryPMAllocation(Metric):
    
    def __call__(self, *inputs) -> float:
        state, pm_selection_constraints = inputs
        vm_cpu_capacity, vm_memory_capacity, vm_used_cpu, vm_used_memory, vm_core = pm_selection_constraints
        
        pm_resources = state.pm_resources
        
        num_pms = len(pm_resources)
        
        vm_wrapped = (vm_cpu_capacity, vm_memory_capacity, vm_core)
        
        selected_pm = -1
        min_observed_diff = float('inf')
        
        for pm_count, pm_stats in enumerate(pm_resources):
            pm_cpu_remaining = pm_stats[0]
            pm_memory_remaining = pm_stats[1]
            pm_core = pm_stats[4]
            
            pm_wrapped = (pm_cpu_remaining, pm_memory_remaining, pm_core)
            
            candidate_diff = pm_memory_remaining - vm_memory_capacity
            
            is_new_min = candidate_diff < min_observed_diff 
            
            if vm_pm_compatible(*pm_wrapped, *vm_wrapped) and is_new_min:
                selected_pm = pm_count
                min_observed_diff = candidate_diff
                
        if selected_pm == -1:
            
            for i, pm_stats in enumerate(AMAZON_PM_TYPES.values):
                
                pm_cpu_remaining, pm_memory_remaining, _, _, pm_cores = pm_stats
                
                pm_wrapped = (pm_cpu_remaining, pm_memory_remaining, pm_cores)
                
                candidate_diff = pm_memory_remaining - vm_memory_capacity
                
                is_new_min = candidate_diff < min_observed_diff 
                
                if vm_pm_compatible(*pm_wrapped, *vm_wrapped) and is_new_min:
                    selected_pm = num_pms + i
                    min_observed_diff = candidate_diff
            
        return { 'pm_num' : selected_pm }