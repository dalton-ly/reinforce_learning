import random
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from grid_world import GridWorld

env = GridWorld(env_size=(5, 5),
                start_state=(0, 0),
                target_state=(2, 3),
                forbidden_states=[(1, 1), (2, 1), (2, 2), (1, 3), (3, 3),(1,4)])
env.reset()
eps=["0.000000","0.100000","0.200000","0.500000"]

for e in eps:
    state_value = np.loadtxt(f"/home/ly/code/cpp/data/state_value_of{e}.csv", delimiter=",").reshape(5, 5)
    direction=np.loadtxt(f"/home/ly/code/cpp/data/best_action_of{e}.csv", delimiter=",").reshape(5, 5)
    flatten_value=state_value.flatten()
    env.add_state_values(flatten_value)
    env.plot_policy(direction)
