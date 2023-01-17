from statistics import mean
import sys
from typing import Dict
import signal
import multiprocessing
import time

import numpy as np
import pandas as pd
import tensorflow as tf
# from keras.utils.generic_utils import get_custom_objects
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings("ignore")

from src.const import *
from src.helpers import *
from src.submodel_preparators import *
from src.training.neural_network import train_neural_network, gen_loss
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def init_worker():
    ''' Add KeyboardInterrupt exception to mutliprocessing workers '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def train_airport_nn(airport, lookahead, Data_Type, MCS, epochs, number_trials, patience, experiment_id, Dir):

    print(f"Beginning {airport} NN Training \n")

    df = pd.read_csv(f"{PROCESSED_DATA_DIR}/{airport}/{airport}_processed_data.csv",index_col=0)
    
    X_train = np.genfromtxt(Dir+f"NN_Data/{Data_Type}/{airport}/{airport}_train_data.csv", delimiter=',')
    X_val = np.genfromtxt(Dir+f"NN_Data/{Data_Type}/{airport}/{airport}_val_data.csv", delimiter=',')
    X_test = np.genfromtxt(Dir+f"NN_Data/{Data_Type}/{airport}/{airport}_test_data.csv", delimiter=',')

    labels = CONFIGS[airport]

    Num_Classes = len(labels.keys())

    start = time.time()

    print("\n"+100*"*"+"\n")
    print(f"Beginning Training")
    print(f"Airport: {airport} | Lookahead: {lookahead}")
    print("\n"+100*"*"+"\n")

    single_lookahead_preparator = SingleConfigDataPreparator(df,airport,lookahead=lookahead)
    
    y_train = single_lookahead_preparator.y_train
    y_val = single_lookahead_preparator.y_val
    y_test = single_lookahead_preparator.y_test

    y_train = return_labels(y_train, labels)
    y_val = return_labels(y_val, labels)
    y_test = return_labels(y_test, labels)

    clf_new = train_neural_network(X_train, to_cat(y_train, Num_Classes),  X_val, to_cat(y_val, Num_Classes), airport, lookahead, MCS, epochs, number_trials, patience, experiment_id, Dir)
    
    best_model = clf_new.export_model()

    end = time.time()
        
    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)

    train_pred = np.clip(train_pred, 1e-8, 1-1e-8)
    val_pred = np.clip(val_pred, 1e-8, 1-1e-8)
    test_pred = np.clip(test_pred, 1e-8, 1-1e-8)

    y_train = to_cat(y_train, Num_Classes)
    y_val   = to_cat(y_val, Num_Classes)
    y_test  = to_cat(y_test, Num_Classes)

    train_loss = log_loss(y_train.flatten(), train_pred.flatten())
    val_loss = log_loss(y_val.flatten(), val_pred.flatten())
    test_loss = log_loss(y_test.flatten(), test_pred.flatten())

    train_time = (end-start)/60

    print("\n"+100*"*"+"\n")
    print(f"Finished Training | {train_time} min")
    print(f"Airport: {airport} | Lookahead: {lookahead} | Train Loss: {train_loss} | Val Loss: {val_loss} | Test Loss: {test_loss}")
    print("\n"+100*"*"+"\n")

    return y_train, y_val, y_test, train_pred, val_pred, test_pred



# def train_final_model(MCS):
#     # pool = multiprocessing.Pool(10, init_worker)

#     # pool.map(train_airport_nn, AIRPORTS)

#     for airport in AIRPORTS:
#         train_airport_nn(airport, MCS)


# if __name__ == "__main__":

#     MCS = True

#     train_final_model(MCS)

    
            


