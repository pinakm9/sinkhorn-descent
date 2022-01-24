"""
Finds a trajectory on the attractor of L96_10
"""
# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, realpath
script_dir = Path(dirname(realpath(__file__)))
module_dir = str(script_dir.parent)
sys.path.insert(0, module_dir + '/modules')

# import remaining modules
import numpy as np
import Lorenz63_xz
import pandas as pd
import matplotlib.pyplot as plt

# set random initial point, load the L96_10 model
np.random.seed(42)
x0 = np.random.uniform(size=3)
model, gen_path = Lorenz63_xz.get_model(x0=x0, size=10, obs_gap=0.2)
length = 500

# find a trajectory on the attractor
total_iters = int(1e5)
batch_size = int(1e4)
for i in range(int(total_iters/batch_size)):
    print('Working on batch #{}'.format(i), end='\r')
    hidden_path = gen_path(x0, batch_size)
    x0 = hidden_path[-1]

path = gen_path(x0, length)
pd.DataFrame(path).to_csv('trajectory_2_{}.csv'.format(length), header=None, index=None)

def check_path(path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(path[:, 0], path[:, 1], path[:, 2])
    plt.show() 

check_path(path)