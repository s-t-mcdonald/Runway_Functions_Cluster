import pandas as pd
import numpy as np
import datetime

from src.const import *
from src.feature_processing_functions import *
from src.helpers import get_training_file_path, get_test_file_path

def create_train_val_test_masks(complete_df,train_timestamps, val_timestamps,test_timestamps):
    complete_df['train_set'] = complete_df.index.isin(train_timestamps)
    complete_df['val_set']  = complete_df.index.isin(val_timestamps)
    complete_df['test_set'] = complete_df.index.isin(test_timestamps)
    return complete_df

def process_airport(airport, date_range, train_timestamps, val_timestamps, test_timestamps):

    processed_time_df          = process_time(date_range)
    df                         = processed_time_df.copy(deep = True)
    # del processed_time_df

    df_dep                          = pd.concat([pd.read_csv(get_training_file_path(airport,DEP_FILE)), pd.read_csv(get_test_file_path(airport,DEP_FILE))])
    processed_dep_rate_df           = process_true_rate(date_range,df_dep,op_type='dep')
    # df = pd.merge_asof(df, processed_dep_rate_df, direction="backward", left_index=True, right_index=True)
    # df                              = df.merge(processed_dep_rate_df,how='left',left_index=True,right_index=True)
    # del processed_dep_rate_df

    df_arr                          = pd.concat([pd.read_csv(get_training_file_path(airport,ARR_FILE)), pd.read_csv(get_test_file_path(airport,ARR_FILE))])
    processed_arr_rate_df           = process_true_rate(date_range,df_arr,op_type='arr')
    # df = pd.merge_asof(df, processed_arr_rate_df, direction="backward", left_index=True, right_index=True)
    # df                              = df.merge(processed_arr_rate_df,how='left',left_index=True,right_index=True)
    # del processed_arr_rate_df

    # df_est_dep = pd.read_csv(get_file_path(airport, EST_DEP_FILE))
    # processed_est_dep_rate_df  = process_projected_rate(date_range,df_est_dep, rate_type='estimated_runway_departure')
    # df = pd.merge_asof(df, processed_est_dep_rate_df, direction="backward", left_index=True, right_index=True)
    # # df = df.merge(processed_est_dep_rate_df,how='left',left_index=True,right_index=True)
    # del processed_est_dep_rate_df

    # df_est_arr = pd.read_csv(get_file_path(airport, EST_ARR_FILE))
    # processed_est_arr_rate_df  = process_projected_rate(date_range,df_est_arr, rate_type='estimated_runway_arrival')
    # df = pd.merge_asof(df, processed_est_arr_rate_df, direction="backward", left_index=True, right_index=True)
    # # df = df.merge(processed_est_arr_rate_df,how='left',left_index=True,right_index=True)
    # del processed_est_arr_rate_df

    # df_sch_arr = pd.read_csv(get_file_path(airport, SCH_ARR_FILE))
    # processed_sch_arr_rate_df  = process_projected_rate(date_range,df_sch_arr, rate_type='scheduled_runway_arrival')
    # df = pd.merge_asof(df, processed_sch_arr_rate_df, direction="backward", left_index=True, right_index=True)
    # # df = df.merge(processed_sch_arr_rate_df,how='left',left_index=True,right_index=True)
    # del processed_sch_arr_rate_dfZ

    df_weather = pd.concat([pd.read_csv(get_training_file_path(airport,WEATHER_FILE)), pd.read_csv(get_test_file_path(airport,WEATHER_FILE))]) 
    processed_weather_df       = process_weather(date_range,df_weather)
    # df = pd.merge_asof(df, processed_weather_df, direction="backward", left_index=True, right_index=True)
    # df = df.merge(processed_weather_df,how='left',left_index=True,right_index=True)
    # del processed_weather_df

    df_config  = pd.concat([pd.read_csv(get_training_file_path(airport,CONFIG_FILE)), pd.read_csv(get_test_file_path(airport,CONFIG_FILE))]) 
    processed_changes_df       = process_config_changes(date_range, df_config,airport = airport)
    processed_rways_df         = process_runways(date_range,df_config,airport = airport)
    
    # df = pd.merge_asof(df, processed_changes_df, direction="backward", left_index=True, right_index=True)
    # df = df.merge(processed_changes_df,how='left',left_index=True,right_index=True)
    # df = pd.merge_asof(df, processed_rways_df, direction="backward", left_index=True, right_index=True)
    # df = df.merge(processed_rways_df,how='left',left_index=True,right_index=True)
    
    # del processed_changes_df
    # del processed_rways_df

    df = pd.concat([processed_time_df,processed_dep_rate_df,processed_arr_rate_df,processed_weather_df,processed_changes_df,processed_rways_df],axis = 1)

    df = create_train_val_test_masks(df,train_timestamps, val_timestamps,test_timestamps)
    df = relabel_configs(df,airport)



    df.to_csv(f"{PROCESSED_DATA_DIR}/{airport}/{airport}_processed_data.csv")


def process_data(train_timestamps, val_timestamps, test_timestamps):

    start_date = datetime.datetime(2020,11,1,0,0,0)
    end_date = datetime.datetime(2022,6,8,0,0,0)
    date_range = (pd.date_range(start=start_date, end=end_date, freq='30T')).to_frame(name = 'timestamp')

    for airport in AIRPORTS:
        
        process_airport(airport, date_range, train_timestamps, val_timestamps, test_timestamps)

        print("finished ", airport)



if __name__ == "__main__":

    train_labels_df = ensure_datetime(pd.read_csv(f"pen_train_labels.csv.bz2"))
    train_timestamps = sorted(list(train_labels_df['timestamp'].unique()))

    open_sub_df = ensure_datetime(pd.read_csv(f"open_submission_format.csv"))
    addval_start_date = datetime.datetime(2021,10,18,10,0,0)
    addval_end_date = datetime.datetime(2021,10,31,16,0,0)
    addval_daterange = (pd.date_range(start=addval_start_date, end=addval_end_date, freq='60T')).to_frame(name = 'timestamp')
    addval_timestamps = sorted(list(addval_daterange['timestamp'].unique()))
    val_timestamps = sorted(list(open_sub_df['timestamp'].unique())) + addval_timestamps


    test_start_date = datetime.datetime(2020,11,1,4,0,0)
    test_end_date = datetime.datetime(2020,11,6,22,0,0)
    test_daterange = (pd.date_range(start=test_start_date, end=test_end_date, freq='60T')).to_frame(name = 'timestamp')
    test_timestamps = sorted(list(test_daterange['timestamp'].unique()))

    process_data(train_timestamps, val_timestamps, test_timestamps)

    
