import pandas as pd
import numpy as np
import sys
import os

from src.const import *
from src.feature_processing_functions import *

from src.nn_pipeline import train_airport_nn

experiment_id = int(sys.argv[1])

Dir = DIR
parameter_array = pd.read_csv(Dir+"parameters/parameter_array.csv").set_index('PARAM')

airport             = parameter_array.loc[experiment_id]["AIRPORTS"]
lookahead           = parameter_array.loc[experiment_id]["LOOKAHEAD"]
data                = parameter_array.loc[experiment_id]["DATA"]
config_support      = parameter_array.loc[experiment_id]["CONFIG_SUPPORT"]
epochs              = parameter_array.loc[experiment_id]["EPOCHS"]
number_trials       = parameter_array.loc[experiment_id]["NUMBER_TRIALS"]
patience            = parameter_array.loc[experiment_id]["PATIENCE"]

os.mkdir(Dir+f"Results/Result_{experiment_id}")
y_train, y_val, y_test, train_pred, val_pred, test_pred = train_airport_nn(airport, lookahead, data, config_support, epochs, number_trials, patience, experiment_id, Dir)

np.savetxt(Dir+f"Results/Result_{experiment_id}/y_train_truth.csv", y_train, delimiter=",")
np.savetxt(Dir+f"Results/Result_{experiment_id}/y_val_truth.csv", y_val, delimiter=",")
np.savetxt(Dir+f"Results/Result_{experiment_id}/y_test_truth.csv", y_test, delimiter=",")

np.savetxt(Dir+f"Results/Result_{experiment_id}/y_train_pred.csv", train_pred, delimiter=",")
np.savetxt(Dir+f"Results/Result_{experiment_id}/y_val_pred.csv", val_pred, delimiter=",")
np.savetxt(Dir+f"Results/Result_{experiment_id}/y_test_pred.csv", test_pred, delimiter=",")

