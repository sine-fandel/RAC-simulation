
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

from deap.algorithms import varAnd, varOr
from env.simulator.code.simulator import simulator

from env.simulator.code.simulator import SimulatorState, Simulator

from env.simulator.code.simulator.io.loading import load_init_env_data, load_container_data, load_os_data,\
                                                    load_test_container_data, load_test_os_data

from env.simulator.code.simulator.metrics.pm import FirstFitPMAllocation

from env.simulator.code.simulator.config import (AMAZON_PM_TYPES,
                                                AMAZON_VM_TYPES,
                                                VM_CPU_OVERHEAD_RATE,
                                                VM_MEMORY_OVERHEAD)

import yaml,os

current_testcase = 0

filename = os.path.join(os.path.dirname(__file__),'config/single_gp.yaml')
config_file = open(filename)
config = yaml.load(config_file, Loader=yaml.FullLoader)

population_size = config["population_size"]
generation_num = config["generation_num"]
cxpb = config["cxpb"]
mutpb = config["mutpb"]
arity = config["arity"]
elitism_size = config["elitism_size"]
tournament_size = config["tournament_size"]
min_depth = config["min_depth"]
max_depth = config["max_depth"]
mut_min_depth = config["mut_min_depth"]
mut_max_depth = config["mut_max_depth"]




def protectedDiv(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


# terminal nodes list
TERMINAL_NODES = {"ARG0": "container_cpu", "ARG1": "container_memories", "ARG2": "remaining_cpu_capacity", "ARG3": "remaining_memory_capacity", "ARG4": "vm_cpu_overhead", "ARG5": "vm_memory_overhead"}
pset = gp.PrimitiveSet("MAIN", arity)
pset.renameArguments(**TERMINAL_NODES)
pset.addPrimitive(np.add, 2)
pset.addPrimitive(np.subtract, 2)
pset.addPrimitive(np.multiply, 2)
pset.addPrimitive(protectedDiv, 2)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=min_depth, max_=max_depth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def eval(individual, sim_state: SimulatorState, containers, os) -> float:
    sim = Simulator(sim_state)
    func = toolbox.compile(expr=individual)
    hist_fitness = []       # save all fitness
    # load the training data
    input_containers = load_container_data(current_testcase)
    input_os = load_os_data(current_testcase)

    for i in range(len(input_os)) :

        action = sim.vm_selection(func, input_containers.iloc[i], input_os.iloc[i]["os-id"])
        
        sim.step_first_layer(action, *tuple(input_containers.iloc[i].to_list()), input_os.iloc[i]["os-id"])
        ff_pm = FirstFitPMAllocation()
        if sim.to_allocate_vm_data != None:
            pm_selection = ff_pm(sim.state, sim.to_allocate_vm_data[1])
            sim.step_second_layer(pm_selection, *tuple(input_containers.iloc[i].to_list()), input_os.iloc[i]["os-id"])

    return sim.running_energy_consumption,
            
toolbox.register("select", tools.selTournament, tournsize=tournament_size)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=mut_min_depth, max_=mut_max_depth)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def set_seed(seed):
    np.random.seed(seed)

def best_individual(population):
    fitness = [ind.fitness for ind in population]
    best_index = fitness.index(max(fitness))
    best_individual = population[best_index]

    return str(best_individual)

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

    # create json file to save training data
    training_result = {
                        "population_size": population_size,
                        "cxpb": cxpb,
                        "mutpb": mutpb,
                        "elitism_size": elitism_size,
                        "tournament_size": tournament_size,
                        "min_depth": min_depth,
                        "max_depth": max_depth,
                        "mut_min_depth": mut_min_depth,
                        "mut_max_depth": mut_max_depth,
                        "generation": {

                        }
                    }
    
    json_file = f"./results/training/run_{run}_core_{config['cpu_num']}.json"
    with open(json_file, "w") as gen_file:
        json.dump(training_result, gen_file)

    
    start_time = time.time()        # overall time consumption
    # Process Pool
    cpu_count = config["cpu_num"]
    print(f"CPU count: {cpu_count}")
    pool = multiprocessing.Pool(cpu_count)      # use multi-process of evaluation

    toolbox.register("map", pool.map)

    pop = toolbox.population(n=population_size)
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
    invalid_ind = [ind for ind in pop]
    # Warm up the simulation
    init_containers, init_os, init_pms, init_pm_types, init_vms, init_vm_types = load_init_env_data(0).values()
    sim_state = SimulatorState(init_pms, init_vms, init_containers, init_os, init_pm_types, init_vm_types)
    # load the training data
    input_containers = load_container_data(0)
    input_os = load_os_data(0)
    toolbox.register("evaluate", eval, sim_state=sim_state, containers=input_containers, os=input_os)
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if hof is not None:
        hof.update(pop)

    record = mstats.compile(pop) if mstats else {}
    record["time"] = (time.time() - start_time) / 60  # Time taken for the generation
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    print("best individual = ", best_individual(pop))

    # update json file to save the best training result of each generation
    new_data = {"0": {"fitnees": logbook.chapters["fitness"].select("min")[-1], "time": logbook.chapters["fitness"].select("time")[-1], "best": best_individual(pop)}}
    with open(json_file, "r") as gen_file:
        data = json.load(gen_file)
        data["generation"].update(new_data)
    with open(json_file, "w") as gen_file:
        json.dump(data, gen_file)
    

    # Begin the generational process
    for gen in range(1, generation_num + 1):
        # Warm up the simulation
        set_seed(gen)
        init_containers, init_os, init_pms, init_pm_types, init_vms, init_vm_types = load_init_env_data(gen).values()
        sim_state = SimulatorState(init_pms, init_vms, init_containers, init_os, init_pm_types, init_vm_types)

        # load the training data
        input_containers = load_container_data(gen)
        input_os = load_os_data(gen)
        toolbox.register("evaluate", eval, sim_state=sim_state, containers=input_containers, os=input_os)
        
        start_training_time = time.time()   # Start time of generation

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
        record["time"] = (time.time() - start_training_time) / 60  # Time taken for the generation
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        print("best individual = ", best_individual(pop))

        # update json file to save the best training result of each generation
        new_data = {str(gen): {"fitnees": logbook.chapters["fitness"].select("min")[-1], "time": logbook.chapters["fitness"].select("time")[-1], "best": best_individual(pop)}}
        with open(json_file, "r") as gen_file:
            data = json.load(gen_file)
            data["generation"].update(new_data)
        with open(json_file, "w") as gen_file:
            json.dump(data, gen_file)
        

    end_time = time.time()
    print("total time = {:}".format((end_time - start_time) / 60))
    print('Best individual : ', str(hof[0]), hof[0].fitness)
    print(hof[0])
    pool.close()
    with open(f'./results/model/pop_{run}.pkl', 'wb') as pop_file:
        pickle.dump(pop, pop_file)

    with open(f'./results/model/log_{run}.pkl', 'wb') as log_file:
        pickle.dump(logbook, log_file)

    with open(f'./results/model/hof_{run}.pkl', 'wb') as hof_file:
        pickle.dump(hof, hof_file)

    # ####################
    # #test
    # ####################

    # def test_eval(individual):
    #     func = toolbox.compile(expr=individual)
    #     hist_fitness = []       # save all fitness


    #     init_containers, init_os, init_pms, init_pm_types, init_vms, init_vm_types = load_init_env_data(0).values()
    #     # Warm up the simulation
    #     sim_state = SimulatorState(init_pms, init_vms, init_containers, init_os, init_pm_types, init_vm_types)
    #     sim = Simulator(sim_state)

    #     # load the training data
    #     input_containers = load_test_container_data(0)
    #     input_os = load_test_os_data(0)

    #     for i in range(len(input_os)) :
    #         # sim.heuristic_method(tuple(input_containers.iloc[i].to_list()), input_os.iloc[i]["os-id"])
    #         action = sim.vm_selection(func, input_containers.iloc[i], input_os.iloc[i]["os-id"])

    #         sim.step_first_layer(action, *tuple(input_containers.iloc[i].to_list()), input_os.iloc[i]["os-id"])
    #         ff_pm = FirstFitPMAllocation()
    #         if sim.to_allocate_vm_data != None:
    #             pm_selection = ff_pm(sim.state, sim.to_allocate_vm_data[1])
    #             sim.step_second_layer(pm_selection, *tuple(input_containers.iloc[i].to_list()), input_os.iloc[i]["os-id"])

    #     return sim.running_energy_unit_time,

    # # save test result for each run
    
    # model = f'./results/model/hof_{run}.pkl'
    # with open(model, 'rb') as file:
    #     hof = pickle.load(file)

    # individual = hof[0]
    # test_result = {
    #                     "population_size": population_size,
    #                     "cxpb": cxpb,
    #                     "mutpb": mutpb,
    #                     "elitism_size": elitism_size,
    #                     "tournament_size": tournament_size,
    #                     "min_depth": min_depth,
    #                     "max_depth": max_depth,
    #                     "mut_min_depth": mut_min_depth,
    #                     "mut_max_depth": mut_max_depth,
    #                     "test_result": test_eval(individual),
    #                 }

    # with open(f"./results/test/run_{run}_core_{cpu_count}.json", "w") as gen_file:
    #     json.dump(test_result, gen_file)

