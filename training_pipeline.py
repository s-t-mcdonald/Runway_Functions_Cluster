import signal
import multiprocessing
import pandas as pd
from run_experiment import run_experiment


def init_worker():
    ''' Add KeyboardInterrupt exception to mutliprocessing workers '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)

param_array = pd.read_csv("parameters/parameter_array.csv")


EXPERIMENT_IDS = param_array.PARAM


pool = multiprocessing.Pool(30, init_worker)
pool.map(run_experiment, EXPERIMENT_IDS)