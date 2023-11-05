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


singlegp = SingleTreeGP()
pop, log, hof = singlegp.perturb()
# singlegp.save(pop, log, hof)
# singlegp.plot_use_pgv(hof)



