import pandas as pd
from typing import Sequence, Tuple, Dict
from functools import wraps
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, auc, roc_curve, log_loss
import json
import multiprocessing
import signal
import os

from src.const import *
from src.submodel_preparators import *


def init_worker():
    ''' Add KeyboardInterrupt exception to mutliprocessing workers '''
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def try_to_load_submodel(airport,model_id):
    try:
        submodel = XGBClassifier()
        submodel.load_model(f"{SUBMODEL_DIR}/{airport}/{model_id}")
        flag = True
    except:
        flag = False

    return flag

def train_submodels(airport):

    df = pd.read_csv(f"{PROCESSED_DATA_DIR}/{airport}/{airport}_processed_data.csv",index_col=0)

    try:
        os.mkdir(f"{SUBMODEL_DIR}/{airport}")
    except:
        None

    k = 1
    for i in range(1,13):
        

        # for dep_rway in DEP_RWAYS[airport]:
        #     model_id = f"submodel_{k}"
        #     flag = try_to_load_submodel(airport,model_id)
        #     if flag:
        #         k = k + 1
        #         continue
        #     model_name = f"BinaryRway_SingleLookahead_{i}_submodel"
        #     binary_rway_preparator = BinaryRwayDataPreparator(df,airport,lookahead=i,rway=dep_rway,operation='departure')
        #     preparator_description = binary_rway_preparator.describe_parameters()
        #     preparator_description['model_name'] = model_name
        #     model = XGBClassifier(objective ='binary:logistic', eval_metric = "auc", verbosity = 0)
        #     model.fit(binary_rway_preparator.train_features_df,binary_rway_preparator.y_train, early_stopping_rounds=10, eval_set=[(binary_rway_preparator.val_features_df,binary_rway_preparator.y_val)])
        #     predictions = model.predict_proba(binary_rway_preparator.val_features_df)
        #     fpr,tpr,thresholds = roc_curve(binary_rway_preparator.val_labels,predictions[:,1])
        #     model.save_model(f"{SUBMODEL_DIR}/{airport}/{model_id}")
        #     with open(f"{SUBMODEL_DIR}/{airport}/{model_id}_description.json", "w") as outfile:
        #         json.dump(preparator_description, outfile)
        #     print(preparator_description['reference'],i,auc(fpr,tpr))
        #     k+=1

        # for arr_rway in ARR_RWAYS[airport]:
        #     model_id = f"submodel_{k}"
        #     flag = try_to_load_submodel(airport,model_id)
        #     if flag:
        #         k = k + 1
        #         continue
        #     model_name = f"BinaryRway_SingleLookahead_{i}_submodel"
        #     binary_rway_preparator = BinaryRwayDataPreparator(df,airport,lookahead=i,rway=arr_rway,operation='arrival')
        #     preparator_description = binary_rway_preparator.describe_parameters()
        #     preparator_description['model_name'] = model_name
        #     model = XGBClassifier(objective ='binary:logistic', eval_metric = "auc", verbosity = 0)
        #     model.fit(binary_rway_preparator.train_features_df,binary_rway_preparator.y_train, early_stopping_rounds=10, eval_set=[(binary_rway_preparator.val_features_df,binary_rway_preparator.y_val)])
        #     predictions = model.predict_proba(binary_rway_preparator.val_features_df)
        #     fpr,tpr,thresholds = roc_curve(binary_rway_preparator.val_labels,predictions[:,1])
        #     model.save_model(f"{SUBMODEL_DIR}/{airport}/{model_id}")
        #     with open(f"{SUBMODEL_DIR}/{airport}/{model_id}_description.json", "w") as outfile:
        #         json.dump(preparator_description, outfile)
        #     print(preparator_description['reference'],i,auc(fpr,tpr))
        #     k+=1
        
        # for rway in RWAYS[airport]:
        #     model_id = f"submodel_{k}"
        #     model_name = f"BinaryRway_SingleLookahead_{i}_submodel"
        #     binary_rway_preparator = BinaryRwayDataPreparator(df,airport,lookahead=i,rway=rway,operation='used',
        #                                                       config_lookback=config_lookback,
        #                                                       changes_lookback=change_lookback,
        #                                                       rway_lookback=rway_lookback)
        #     preparator_description = binary_rway_preparator.describe_parameters()
        #     preparator_description['model_name'] = model_name
        #     model = XGBClassifier(objective ='binary:logistic', eval_metric = "auc", verbosity = 0)
        #     model.fit(binary_rway_preparator.train_features_df,binary_rway_preparator.y_train, early_stopping_rounds=10, eval_set=[(binary_rway_preparator.val_features_df,binary_rway_preparator.y_val)])
        #     predictions = model.predict_proba(binary_rway_preparator.val_features_df)
        #     fpr,tpr,thresholds = roc_curve(binary_rway_preparator.val_labels,predictions[:,1])
        #     model.save_model(f"{SUBMODEL_DIR}/{airport}/{model_id}")
        #     with open(f"{SUBMODEL_DIR}/{airport}/{model_id}_description.json", "w") as outfile:
        #         json.dump(preparator_description, outfile)
        #     print(preparator_description['reference'],i,auc(fpr,tpr))
        #     k+=1




        for c in list(CONFIGS[airport].keys())[:4]:
            model_id = f"submodel_{k}"
            flag = try_to_load_submodel(airport,model_id)
            if flag:
                k = k + 1
                continue
            model_name = f"BinaryConfig_SingleLookahead_{i}_submodel"
            binary_config_preparator = BinaryConfigDataPreparator(df,airport,lookahead=i,configuration=c)
            preparator_description = binary_config_preparator.describe_parameters()
            preparator_description['model_name'] = model_name
            model = XGBClassifier(objective ='binary:logistic', eval_metric = "auc", verbosity = 0)
            model.fit(binary_config_preparator.train_features_df,binary_config_preparator.y_train, early_stopping_rounds=10, eval_set=[(binary_config_preparator.val_features_df,binary_config_preparator.y_val)])
            predictions = model.predict_proba(binary_config_preparator.val_features_df)
            fpr,tpr,thresholds = roc_curve(binary_config_preparator.val_labels,predictions[:,1])
            model.save_model(f"{SUBMODEL_DIR}/{airport}/{model_id}")
            with open(f"{SUBMODEL_DIR}/{airport}/{model_id}_description.json", "w") as outfile:
                json.dump(preparator_description, outfile)
            print(preparator_description['configuration'],i,auc(fpr,tpr))
            k+=1
        
        
        # model_id = f"submodel_{k}"
        # model_name = f'AllConfig_SingleLookahead_{i}_submodel'
        # single_lookahead_preparator = SingleConfigDataPreparator(df,airport,lookahead=i)
        # preparator_description = single_lookahead_preparator.describe_parameters()
        # preparator_description["model_name"] = model_name
        # model = XGBClassifier(objective = 'multi:softprob', eval_metric = "mlogloss", verbosity = 0)
        # model.fit(single_lookahead_preparator.train_features_df,single_lookahead_preparator.y_train, early_stopping_rounds=10, eval_set=[(single_lookahead_preparator.val_features_df,single_lookahead_preparator.y_val)])
        # predictions = model.predict(single_lookahead_preparator.val_features_df)
        # accuracy = accuracy_score(single_lookahead_preparator.val_labels,predictions)
        # model.save_model(f"{SUBMODEL_DIR}/{airport}/{model_id}")
        # with open(f"{SUBMODEL_DIR}/{airport}/{model_id}_description.json", "w") as outfile:
        #     json.dump(preparator_description, outfile)
        # k += 1
        # print(i, accuracy)

        model_id = f"submodel_{k}"
        flag = try_to_load_submodel(airport,model_id)
        if flag:
            k = k + 1
            continue
        model_name = f'Change_SingleLookahead_{i}_submodel'
        change_single_preparator = ChangeDataPreparator(df,airport,lookahead=i)
        preparator_description = change_single_preparator.describe_parameters()
        preparator_description["model_name"] = model_name
        model = XGBClassifier(objective ='binary:logistic', eval_metric = "auc", verbosity = 0)
        model.fit(change_single_preparator.train_features_df,change_single_preparator.y_train, early_stopping_rounds=10, eval_set=[(change_single_preparator.val_features_df,change_single_preparator.y_val)])
        predictions = model.predict_proba(change_single_preparator.val_features_df)
        fpr,tpr,thresholds = roc_curve(change_single_preparator.val_labels,predictions[:,1])
        model.save_model(f"{SUBMODEL_DIR}/{airport}/{model_id}")
        with open(f"{SUBMODEL_DIR}/{airport}/{model_id}_description.json", "w") as outfile:
            json.dump(preparator_description, outfile)
        k += 1
        print(i,auc(fpr,tpr))
        
    
    # config_buckets = [(1,3),(4,6),(7,9),(10,12),(1,6),(7,12),(1,12)]

    # for low,high in config_buckets:

    #     model_id = f"submodel_{k}"
    #     model_name = f'AllConfig_AggregateLookahead_{low}_{high}_submodel'
    #     aggregate_config_preparator = AggregateConfigDataPreparator(df,airport,lookahead_low_idx = low,lookahead_high_idx =high, decide_based_on =0)
    #     preparator_description = aggregate_config_preparator.describe_parameters()
    #     preparator_description["model_name"] = model_name
    #     model = XGBClassifier(objective = 'multi:softprob', eval_metric = "mlogloss", verbosity = 0)
    #     model.fit(aggregate_config_preparator.train_features_df,aggregate_config_preparator.y_train, early_stopping_rounds=10, eval_set=[(aggregate_config_preparator.val_features_df,aggregate_config_preparator.y_val)])
    #     predictions = model.predict(aggregate_config_preparator.val_features_df)
    #     accuracy = accuracy_score(aggregate_config_preparator.val_labels,predictions)
    #     model.save_model(f"{SUBMODEL_DIR}/{airport}/{model_id}")
    #     with open(f"{SUBMODEL_DIR}/{airport}/{model_id}_description.json", "w") as outfile:
    #         json.dump(preparator_description, outfile)
    #     k += 1
    #     print(low,high, accuracy)
    
    # change_buckets = [(1,4),(4,7),(7,10),(10,13),(1,7),(7,13),(1,13)]

    # for low,high in change_buckets:

    #     model_id = f"submodel_{k}"
    #     model_name = f'Change_AggregateLookahead_{i}_{low}_{high}_submodel'
    #     aggregate_change_preparator = AggregateChangeDataPreparator(df,airport,lookahead_low_idx = low,lookahead_high_idx =high)
    #     preparator_description = aggregate_change_preparator.describe_parameters()
    #     preparator_description["model_name"] = model_name
    #     model = XGBClassifier(objective ='binary:logistic', eval_metric = "auc", verbosity = 0)
    #     model.fit(aggregate_change_preparator.train_features_df,aggregate_change_preparator.y_train, early_stopping_rounds=10, eval_set=[(aggregate_change_preparator.val_features_df,aggregate_change_preparator.y_val)])
    #     predictions = model.predict_proba(aggregate_change_preparator.val_features_df)
    #     fpr,tpr,thresholds = roc_curve(aggregate_change_preparator.val_labels,predictions[:,1])
    #     model.save_model(f"{SUBMODEL_DIR}/{airport}/{model_id}")
    #     with open(f"{SUBMODEL_DIR}/{airport}/{model_id}_description.json", "w") as outfile:
    #         json.dump(preparator_description, outfile)
    #     k += 1
    #     print(low,high,auc(fpr,tpr))

    # print("\n"+100*"-"+"\n")
    # print("Finished Airport: " + airport)
    # print("\n"+100*"-"+"\n")

def train_all_submodels():

    for airport in AIRPORTS:
        train_submodels(airport)

    # pool = multiprocessing.Pool(10, init_worker)
    # pool.map(train_submodels, AIRPORTS)
    # train_submodels()

    

if __name__ == "__main__":

    train_all_submodels()

    

    