import sys
import os

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.const import *
from src.feature_processing_functions import *

from src.nn_pipeline import train_airport_nn

def run_experiment(experiment_id):

    Dir = DIR
    parameter_array = pd.read_csv(Dir+"parameters/parameter_array.csv").set_index('PARAM')

    airport             = parameter_array.loc[experiment_id]["AIRPORTS"]
    lookahead           = parameter_array.loc[experiment_id]["LOOKAHEAD"]
    data                = parameter_array.loc[experiment_id]["DATA"]
    config_support      = parameter_array.loc[experiment_id]["CONFIG_SUPPORT"]
    epochs              = parameter_array.loc[experiment_id]["EPOCHS"]
    number_trials       = parameter_array.loc[experiment_id]["NUMBER_TRIALS"]
    patience            = parameter_array.loc[experiment_id]["PATIENCE"]

    try:
        os.mkdir(Dir+f"Results/Result_{experiment_id}")
    except:
        None

    try:
        pd.read_csv(Dir+f"Results/Result_{experiment_id}/results_df.csv")
    except:
        print("File Exists\n")
        return None
        
    y_train, y_val, y_test, train_pred, val_pred, test_pred = train_airport_nn(airport, lookahead, data, config_support, epochs, number_trials, patience, experiment_id, Dir)


    train_loss = log_loss(y_train.flatten(), train_pred.flatten())
    val_loss = log_loss(y_val.flatten(), val_pred.flatten())
    test_loss = log_loss(y_test.flatten(), test_pred.flatten())

    train_accuracy = accuracy_score(np.argmax(y_train,axis=1), np.argmax(train_pred,axis=1))
    val_accuracy = accuracy_score(np.argmax(y_val,axis=1), np.argmax(val_pred,axis=1))
    test_accuracy = accuracy_score(np.argmax(y_test,axis=1), np.argmax(test_pred,axis=1))

    train_f1 = f1_score(np.argmax(y_train,axis=1), np.argmax(train_pred,axis=1), average='weighted')
    val_f1 = f1_score(np.argmax(y_val,axis=1), np.argmax(val_pred,axis=1), average='weighted')
    test_f1 = f1_score(np.argmax(y_test,axis=1), np.argmax(test_pred,axis=1), average='weighted')

    df_dict = {"param": [experiment_id], "train_loss": [train_loss], "val_loss": [val_loss], "test_loss": [test_loss],
                    "train_accuracy": [train_accuracy], "val_accuracy": [val_accuracy], "test_accuracy": [test_accuracy],
                    "train_f1": [train_f1], "val_f1": [val_f1], "test_f1": [test_f1]}

    df = pd.DataFrame(df_dict)

    df.to_csv(Dir+f"Results/Result_{experiment_id}/results_df.csv")

if __name__ == "__main__":

    experiment_id = int(sys.argv[1])
    run_experiment(experiment_id)

# np.savetxt(Dir+f"Results/Result_{experiment_id}/y_train_truth.csv", y_train, delimiter=",")
# np.savetxt(Dir+f"Results/Result_{experiment_id}/y_val_truth.csv", y_val, delimiter=",")
# np.savetxt(Dir+f"Results/Result_{experiment_id}/y_test_truth.csv", y_test, delimiter=",")

# np.savetxt(Dir+f"Results/Result_{experiment_id}/y_train_pred.csv", train_pred, delimiter=",")
# np.savetxt(Dir+f"Results/Result_{experiment_id}/y_val_pred.csv", val_pred, delimiter=",")
# np.savetxt(Dir+f"Results/Result_{experiment_id}/y_test_pred.csv", test_pred, delimiter=",")

