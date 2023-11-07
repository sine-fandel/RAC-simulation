
import random
import numpy as np
import multiprocessing
# from config.base_config import BaseConfig
import operator
import math
import random
import pickle
import time

# from builder import Builder
from deap import base
from deap import creator
from deap import tools
from deap import gp

from optim.multi_tree_gp import MultiPrimitiveTree, cxOnePoint_type_wise, mutUniform_multi_tree, staticLimit

from deap.algorithms import varAnd
from env.simulator.code.simulator import simulator

from env.simulator.code.simulator import SimulatorState, Simulator

from env.simulator.code.simulator.io.loading import load_init_env_data, load_container_data, load_os_data,\
                                                    load_test_container_data, load_test_os_data

from env.simulator.code.simulator.metrics.pm import FirstFitPMAllocation

import yaml,os


filename = os.path.join(os.path.dirname(__file__),'config/dual_gp.yaml')
config_file = open(filename)
config = yaml.load(config_file, Loader=yaml.FullLoader)

sub_population_size0 = config["sub_population_size0"]
sub_population_size1 = config["sub_population_size1"]
generation_num = config["generation_num"]
cxpb = config["cxpb"]
mutpb = config["mutpb"]
arity0 = config["arity0"]
arity1 = config["arity1"]
elitism_size = config["elitism_size"]
tournament_size = config["tournament_size"]
min_depth = config["min_depth"]
max_depth = config["max_depth"]
mut_min_depth = config["mut_min_depth"]
mut_max_depth = config["mut_max_depth"]
testcase_num = config["testcase_num"]


def protectedDiv(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x

pset = {"vm": None, "pm": None}
for type, item in pset.items():
    if type == "vm":
        pset[type] = gp.PrimitiveSet(type, arity0)
    else:
        pset[type] = gp.PrimitiveSet(type, arity1)
    pset[type].addPrimitive(np.add, 2)
    pset[type].addPrimitive(np.subtract, 2)
    pset[type].addPrimitive(np.multiply, 2)
    pset[type].addPrimitive(protectedDiv, 2)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", MultiPrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# register two trees as one individual
toolbox.register(
     "expr",
    lambda: {
        "vm":
            gp.genHalfAndHalf(
                pset=pset["vm"],
                min_=min_depth,
                max_=max_depth
            ),
        "pm":
            gp.genHalfAndHalf(
                pset=pset["pm"],
                min_=min_depth,
                max_=max_depth
            )

    }
)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def eval(individual):
    """evaluate dual-tree gp individual 
    """
    func0 = toolbox.compile(expr=individual["vm"], pset=pset["vm"])
    func1 = toolbox.compile(expr=individual["pm"], pset=pset["pm"])
    hist_fitness = []       # save all fitness

    print("VM selection polciy: ", str(individual["vm"]))
    print("PM selection polciy: ", str(individual["pm"]))

    test_case_num = testcase_num

    for case in range(test_case_num):
        init_containers, init_os, init_pms, init_pm_types, init_vms, init_vm_types = load_init_env_data(case).values()
        # Warm up the simulation
        sim_state = SimulatorState(init_pms, init_vms, init_containers, init_os, init_pm_types, init_vm_types)
        sim = Simulator(sim_state)

        # load the training data
        input_containers = load_container_data(case)
        input_os = load_os_data(case)
        for i in range(len(input_os)) :
            # sim.heuristic_method(tuple(input_containers.iloc[i].to_list()), input_os.iloc[i]["os-id"])
            candidates = sim.vm_candidates(tuple(input_containers.iloc[i].to_list()), input_os.iloc[i]["os-id"])
            largest_priority = float("inf")
            selected_vm = -1
            for vm in candidates:
                if largest_priority > func0(*tuple(vm[ : -1])):
                    selected_vm = vm[-1]
                    largest_priority = func0(*vm[ : -1])

            action = {"vm_num": selected_vm}
            sim.step_first_layer(action, *tuple(input_containers.iloc[i].to_list()), input_os.iloc[i]["os-id"])
            if sim.to_allocate_vm_data != None:
                largest_priority = float("inf")
                selected_pm = -1
                pm_candidates = sim.pm_candidates()
                for pm in pm_candidates:
                    if largest_priority > func1(*tuple(pm[ : -1])):
                        selected_pm = pm[-1]
                        largest_priority = func1(*pm[ : -1])

                action = {"pm_num": selected_pm}
                sim.step_second_layer(action, *tuple(input_containers.iloc[i].to_list()), input_os.iloc[i]["os-id"])


        hist_fitness.append(sim.running_energy_unit_time)

    print(math.fsum(hist_fitness) / test_case_num)
    return math.fsum(hist_fitness) / test_case_num,
            

toolbox.register("evaluate", eval)
toolbox.register("select", tools.selTournament, tournsize=tournament_size)
toolbox.register("mate", cxOnePoint_type_wise)
toolbox.register("expr_mut", gp.genFull, min_=mut_min_depth, max_=mut_max_depth)
toolbox.register("mutate", mutUniform_multi_tree, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", staticLimit(key=operator.attrgetter("height"), max_value=17))


def set_seed(seed):
    random.seed(seed)

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    import json

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--RUN", help="run", dest="run", type=int, default="0")
    parser.add_argument("-s", "--SEED", help="seed", dest="seed", type=int, default="0")
    args = parser.parse_args()

    set_seed(args.seed)
    run = args.run

    start_time = time.time()
    # Process Pool
    cpu_count = config["cpu_num"]
    print(f"CPU count: {cpu_count}")
    pool = multiprocessing.Pool(cpu_count)      # use multi-process of evaluation

    toolbox.register("map", pool.map)

    pop = toolbox.population(n=sub_population_size0)
    hof = tools.HallOfFame(elitism_size)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    verbose = True
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (mstats.fields if mstats else [])
    start_time = time.time()  # Start time of generation

    # Evaluate the individuals with an invalid fitness
    start_time = time.time()  
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if hof is not None:
        hof.update(pop)

    record = mstats.compile(pop) if mstats else {}
    record["time"] = time.time() - start_time  # Time taken for the generation
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    
    # Begin the generational process
    for gen in range(1, generation_num + 1):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if hof is not None:
            hof.update(offspring)

        # Replace the current population by the offspring
        pop[:] = offspring

        # Append the current generation statistics to the logbook
        record = mstats.compile(pop) if mstats else {}
        record["time"] = time.time() - start_time  # Time taken for the generation
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        end_training_time = time.time()   # Start time of generation
        

    end_time = time.time()
    print("total time = {:}".format(end_time - start_time))
    print('Best individual : ', str(hof[0]), hof[0].fitness)

    pool.close()
