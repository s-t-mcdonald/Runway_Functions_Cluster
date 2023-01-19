from email.policy import default
import tensorflow as tf
import autokeras as ak
from tensorflow import keras
from tensorflow import nest
from tensorflow.keras import activations
from tensorflow.keras.layers import Dropout, Dense, Activation, Softmax
from autokeras.blocks import reduction
from autokeras.utils import utils
from keras.utils.generic_utils import get_custom_objects
import keras
import numpy as np
import pandas as pd

from src.const import *


def past_final_layer(airport, lookahead):

    class final_layer(ak.ClassificationHead):
        def build(self, hp, inputs=None):

            features = inputs[1]
            inputs = inputs[0]

            # Get the input_node from inputs.
            inputs = nest.flatten(inputs)
            utils.validate_num_inputs(inputs, 1)
            input_node = inputs[0]
            output_node = input_node

            # Reduce the tensor to a vector.
            if len(output_node.shape) > 2:
                output_node = reduction.SpatialReduction().build(hp, output_node)

            if self.dropout is not None:
                dropout = self.dropout
            else:
                dropout = hp.Choice("dropout", [0.0, 0.25, 0.5], default=0.5)

            if dropout > 0:
                output_node = Dropout(dropout)(output_node)
            output_node = Dense(self.shape[-1])(output_node)
            if isinstance(self.loss, keras.losses.BinaryCrossentropy):
                output_node = Activation(activations.sigmoid, name=self.name)(
                    output_node
                )
            else:
                output_node = Softmax(name=self.name)(output_node)

            min_support = hp.Choice("min_support", [0.0001], default=0.0001)
            config_support = hp.Float("config_support", min_value=0.1, max_value=0.97, default=0.8*CONFIG_SUPPORT_DEFAULTS[airport][lookahead])

            ## Miniumum Support and Minimum Config Support ##
            y_pred = output_node
            Num_Class = y_pred.shape[1]
            configs = features[:,-Num_Class:]

            y_pred = y_pred*(1-config_support)

            # Min Config Support
            y_pred = y_pred + config_support*configs

            # Min Support
            y_pred = y_pred*(1-min_support*y_pred.shape[1])
            y_pred = y_pred + min_support

            # Return Predictions
            return y_pred

    return final_layer

def gen_loss():

    def binary_loss(y_true, y_pred):

        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(y_true, [-1])

        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        return bce(y_true, y_pred)

    return binary_loss


def train_neural_network(X_train, y_train, X_val, y_val, airport, lookahead, MCS, epochs, number_trials, patience, experiment_id, Dir):

    cbs = [
        tf.keras.callbacks.EarlyStopping(patience=patience)
    ]

    binary_loss = gen_loss()

    get_custom_objects().update({"binary_loss": binary_loss})

    final_layer = past_final_layer(airport, lookahead)

    ## Autokeras Training Routine ##
    input_node = ak.Input()
    output_node = ak.Normalization()(input_node)
    output_node = ak.DenseBlock(num_layers=2,num_units=256,use_batchnorm=False,dropout=0.5)(output_node)

    if MCS > 0.5:
        print("Using MCS\n")
        output_node = final_layer(dropout=0.1)([output_node, input_node])
    else:
        print("Not Using MCS\n")
        output_node = ak.ClassificationHead(dropout=0.1)(output_node)

    project_name = Dir + f"ToDelete/Result_{experiment_id}/automodel"
    clf = ak.AutoModel(
        project_name=project_name, inputs=input_node, 
        outputs=output_node, loss=binary_loss, overwrite=True, max_trials=number_trials, optimizer="adam", tuner="hyperband"
    )

    # Train Model and Fit Best Hyperparemters
    clf.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=1, callbacks=cbs)

    try:
        opt_config_support = clf.tuner.get_best_hyperparameters()[0].values["final_layer_1/config_support"]
    except:
        opt_config_support = -1
    
    norm_support = CONFIG_SUPPORT_DEFAULTS[airport][lookahead]




    return clf, norm_support, opt_config_support
