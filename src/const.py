DIR = "/home/mcdonst/Runway_Functions_Cluster/"
TRAINING_DATA_DIR     = DIR+"training_data"
TEST_DATA_DIR     = DIR+"test_data"
PROCESSED_DATA_DIR     = DIR+"processed_data"
MODEL_DIR = DIR+"trained_models"
SUBMODEL_DIR = DIR+"trained_submodels"

"""THE BELOW ARE IN THEIR RESPECTIVE AIRPORT DIRECTORY WITHIN DATA_DIR"""

CONFIG_FILE    = "airport_config.csv.bz2"
ARR_FILE       = "arrival_runway.csv.bz2"
DEP_FILE       = "departure_runway.csv.bz2"
EST_DEP_FILE   = "etd.csv.bz2"
FIRST_POS_FILE = "first_position.csv.bz2"
WEATHER_FILE   = "lamp.csv.bz2"
RWAY_ARR_FILE  = "mfs_runway_arrival_time.csv.bz2"
RWAY_DEP_FILE  = "mfs_runway_departure_time"
STAND_ARR_FILE = "mfs_stand_arrival_time.csv.bz2"
STAND_DEP_FILE = "mfs_stand_departure_time.csv.bz2"
SCH_ARR_FILE   = "tbfm_scheduled_runway_arrival_time.csv.bz2"
EST_ARR_FILE   = "tfm_estimated_runway_arrival_time.csv.bz2"

ALL_FILES = [CONFIG_FILE, ARR_FILE, DEP_FILE, EST_DEP_FILE,
            FIRST_POS_FILE, WEATHER_FILE, RWAY_ARR_FILE, RWAY_DEP_FILE,
            STAND_ARR_FILE, STAND_DEP_FILE, SCH_ARR_FILE,EST_ARR_FILE]


DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

DATETIME_COLS = ['timestamp', 'scheduled_runway_arrival_time','estimated_runway_departure_time','estimated_runway_arrival_time','forecast_timestamp']
"""THE BELOW ARE IN DATA_DIR AND CONTAIN ALL AIRPORTS"""

TRAIN_LABELS_FILE = "open_train_labels.csv.bz2"
SUBMISSION_FILE   = "open_submission_format.csv"

AIRPORTS = [
    "katl",
    "kclt", 
    "kden",
    "kdfw",
    "kjfk",
    "kmem",
    "kmia",
    "kord",
    "kphx",
    "ksea"
]


DEP_RWAYS = {'katl':{'26R', '10', '8R', '27R', '27L', '9L', '28', '9R', '26L', '8L'},
             'kclt':{'36R', '18L', '36L', '36C', '18C', '18R'},
             'kden':{'17L', '8', '26', '25', '34L', '16L', '35L', '17R', '34R'},
             'kdfw':{'36R', '31L', '17L', '31R', '18L', '36L', '35C', '17C', '18R', '35L', '17R'},
             'kjfk':{'31L', '31R', '13R', '4L', '4R', '13L', '22R'},
             'kmem':{'36R', '18L', '36C', '27', '36L', '18C', '18R', '9'},
             'kmia':{'30', '26R', '27', '8R', '12', '9', '26L', '8L'},
             'kord':{'28R', '9R', '22L', '27C', '27L', '4L', '10L', '4R', '28C', '9C', '10C'},
             'kphx':{'7R', '8', '25R', '25L', '26', '7L'},
             'ksea':{'16C', '34C', '34R', '16L'}}

ARR_RWAYS = {'katl':{'26R', '10', '27R', '27L', '8R', '9L', '28', '9R', '26L', '8L'},
             'kclt':{'36R', '18L', '36L', '36C', '18C', '18R'},
             'kden':{'17L', '8', '35R', '26', '16L', '7', '34L', '25', '35L', '16R', '17R', '34R'},
             'kdfw':{'36R', '17L', '31L', '13L', '35R', '31R', '18L', '13R', '36L', '35C', '17C', '18R', '35L', '17R'},
             'kjfk':{'31L', '31R', '22L', '13R', '4L', '4R', '13L', '22R'},
             'kmem':{'36R', '18L', '36C', '27', '36L', '18C', '18R', '9'},
             'kmia':{'30', '26R', '27', '8R', '12', '9', '26L', '8L'},
             'kord':{'28L', '28R', '9R', '22L', '27C', '27R', '27L', '10L', '4R', '9L', '9C', '28C', '10R', '22R', '10C'},
             'kphx':{'7R', '8', '25R', '25L', '26', '7L'},
             'ksea':{'34C', '16C', '16L', '34L', '16R', '34R'}}


RWAYS     = {'katl':{'26R', '10', '27R', '27L', '8R', '9L', '28', '9R', '26L', '8L'},
             'kclt':{'36R', '18L', '36L', '36C', '18C', '18R'},
             'kden':{'17L', '8', '35R', '26', '16L', '7', '34L', '25', '35L', '16R', '17R', '34R'},
             'kdfw':{'36R', '17L', '31L', '13L', '35R', '31R', '18L', '13R', '36L', '35C', '17C', '18R', '35L', '17R'},
             'kjfk':{'31L', '31R', '22L', '13R', '4L', '4R', '13L', '22R'},
             'kmem':{'36R', '18L', '36C', '27', '36L', '18C', '18R', '9'},
             'kmia':{'30', '26R', '27', '8R', '12', '9', '26L', '8L'},
             'kord':{'28L', '28R', '27R', '28C', '9R', '27C', '9L', '22R', '10L', '22L', '10C', '4R', '9C', '27L', '10R', '4L'},
             'kphx':{'7R', '8', '25R', '25L', '26', '7L'},
             'ksea':{'34C', '16C', '16L', '34L', '16R', '34R'}}



CONFIGS   = {'katl': {'D_26L_27R_A_26R_27L_28': 0, 'D_8R_9L_A_10_8L_9R': 1, 'other': 2, 'D_26L_27R_28_A_26R_27L_28': 3, 'D_26L_27R_A_26L_27L_28': 4, 'D_8R_9R_A_10_8L_9R': 5, 'D_26L_28_A_26L_28': 6, 'D_26R_28_A_26R_28': 7, 'D_26L_27R_A_26R_27L': 8, 'D_26L_27L_A_26R_27L_28': 9, 'D_26L_27R_A_26L_27R_28': 10, 'D_26L_27R_A_26R_27R_28': 11, 'D_26R_27R_A_26R_27L_28': 12, 'D_26L_27R_A_27L_28': 13, 'D_10_8R_A_10_8R': 14, 'D_8R_9L_A_8L_9R': 15, 'D_26L_28_A_26R_28': 16, 'D_26L_28_A_26R_27L_28': 17, 'D_8R_9L_A_10_8R_9R': 18, 'D_10_8R_9L_A_10_8L_9R': 19, 'D_26L_27R_A_26R_28': 20, 'D_10_8L_A_10_8L': 21, 'D_8R_9L_A_10_9R': 22, 'D_8L_9L_A_10_8L_9R': 23, 'D_8R_9L_A_8R_9L': 24, 'D_26L_27R_A_26L_27R': 25, 'D_9L_A_9R': 26},
             'kclt': {'D_36C_36R_A_36C_36L_36R': 0, 'D_18C_18L_A_18C_18L_18R': 1, 'D_36C_36R_A_36C_36R': 2, 'D_36R_A_36R': 3, 'D_18C_18L_A_18C_18L': 4, 'D_18L_A_18L': 5, 'D_36R_A_36L_36R': 6, 'other': 7, 'D_36C_A_36C': 8, 'D_18C_A_18C': 9, 'D_18L_A_18L_18R': 10, 'D_36C_A_36C_36L': 11, 'D_18C_A_18C_18R': 12},
             'kden': {'other': 0, 'D_25_34L_8_A_34R_35L_35R': 1, 'D_17L_25_8_A_16L_16R_17R': 2, 'D_25_34L_A_34R_35L_35R': 3, 'D_25_34L_8_A_35L_35R': 4, 'D_17L_8_A_16R_17R': 5, 'D_17L_25_8_A_16R_17R': 6, 'D_17L_17R_8_A_16L_16R_17R': 7, 'D_17L_8_A_16L_16R_17R': 8, 'D_25_34L_A_26_35L_35R': 9, 'D_34L_A_35L_35R': 10, 'D_34L_A_34R_35L_35R': 11, 'D_17R_A_16L_16R_17R': 12, 'D_25_34L_A_35L_35R': 13, 'D_34L_34R_A_35L_35R': 14, 'D_34L_8_A_35L_35R_7': 15, 'D_34L_34R_A_34R_35L_35R': 16, 'D_17R_8_A_16L_16R_17R': 17, 'D_34L_8_A_34R_35L_35R': 18, 'D_17L_8_A_16R_17R_7': 19, 'D_17R_A_16R_17R': 20, 'D_17R_8_A_16R_17R': 21, 'D_17R_A_16L_17R': 22, 'D_34L_8_A_35L_35R': 23, 'D_25_34L_34R_A_34R_35L_35R': 24, 'D_17L_17R_8_A_16R_17R': 25, 'D_17R_A_17R': 26, 'D_17L_17R_25_8_A_16L_16R_17R': 27, 'D_25_8_A_16R_35L_35R': 28, 'D_17L_17R_A_16L_16R_17R': 29, 'D_34L_A_35L': 30, 'D_34R_A_35L': 31, 'D_25_34L_34R_8_A_34R_35L_35R': 32, 'D_34L_34R_8_A_34R_35L_35R': 33, 'D_17L_8_A_17R': 34, 'D_17L_17R_A_16R_17R': 35, 'D_17L_25_A_16L_16R_17R': 36, 'D_17R_8_A_16L_17R': 37, 'D_34R_A_34R_35L_35R': 38, 'D_25_34L_34R_A_26_35L_35R': 39, 'D_17R_8_A_17R': 40, 'D_17L_17R_8_A_17R': 41},
             'kdfw': {'D_17R_18L_A_13R_17C_17L_18R': 0, 'other': 1, 'D_17R_18L_A_13R_17C_17L_18L': 2, 'D_35L_36R_A_31R_35C_35R_36L': 3, 'D_17R_18L_A_13R_17C_17L': 4, 'D_35L_36R_A_31R_35C_35R': 5, 'D_17R_18L_A_17C_17R_18L': 6, 'D_17R_18L_A_17C_17R_18L_18R': 7, 'D_31L_35L_36R_A_31R_35C_35R_36R': 8, 'D_35L_36R_A_31R_35C_35R_36R': 9, 'D_35L_36R_A_35C_35L_36R': 10, 'D_17R_18L_A_17R_18L': 11, 'D_17C_18L_A_17C_18L': 12, 'D_17R_18L_A_17C_18L': 13, 'D_35L_36R_A_35C_36R': 14, 'D_17R_18L_A_17C_18R': 15, 'D_35L_36R_A_35C_35L_36L_36R': 16, 'D_35L_36R_A_35C_35R_36L': 17, 'D_17R_18L_A_17C_17L_18R': 18, 'D_35L_A_35C': 19, 'D_31L_35L_36R_A_31R_35C_35R': 20, 'D_35L_36R_A_35C_35R_36R': 21, 'D_17R_18L_A_17C_17L_18L': 22, 'D_35C_36R_A_35C_36R': 23, 'D_17R_A_17C_17R': 24, 'D_17C_A_17C': 25, 'D_17R_A_17C': 26, 'D_35L_A_35L': 27, 'D_18L_A_18L': 28, 'D_17R_18L_A_17R_18L_18R': 29, 'D_31L_35L_A_31R_35C_35R_36R': 30},
             'kjfk': {'D_31L_A_31L_31R': 0, 'D_22R_A_22L_22R': 1, 'D_4L_A_4L_4R': 2, 'D_13R_A_13L_22L': 3, 'D_13R_A_13L': 4, 'D_22R_A_22L': 5, 'other': 6, 'D_31L_A_31L': 7, 'D_4L_A_4L': 8, 'D_22R_31L_A_22L_22R': 9, 'D_31L_4L_A_4L_4R': 10, 'D_31L_A_31R': 11, 'D_31L_31R_A_31L_31R': 12, 'D_13R_A_22L': 13},
             'kmem': {'D_27_36C_36L_36R_A_27_36L_36R': 0, 'D_18C_18L_18R_A_18L_18R_27': 1, 'D_18C_18L_18R_A_18C': 2, 'other': 3, 'D_18C_18L_18R_A_18L_18R': 4, 'D_18C_18L_18R_27_A_27': 5, 'D_36C_36L_36R_A_36C': 6, 'D_36C_36L_36R_A_36L_36R': 7, 'D_36C_A_27_36L_36R': 8, 'D_18C_18L_18R_A_18C_18L_18R_27': 9, 'D_27_36C_36L_36R_A_27_36C_36L_36R': 10, 'D_36C_A_36L_36R_9': 11, 'D_18C_A_18L_18R_27': 12, 'D_36C_36L_36R_9_A_36L_36R_9': 13, 'D_18C_18L_18R_27_A_18C_18L_18R_27': 14, 'D_18C_A_18L_18R': 15, 'D_36C_A_36L_36R': 16, 'D_27_36L_36R_A_27_36L_36R': 17, 'D_27_36C_36L_A_27_36C_36L': 18, 'D_18C_18R_A_18C_18R_27': 19, 'D_18C_18L_A_18L_27': 20, 'D_18L_18R_A_18L_18R_27': 21, 'D_18C_18L_18R_A_18C_18L_18R': 22, 'D_36C_36L_A_36C_36L': 23, 'D_36C_36L_36R_A_27_36L_36R': 24, 'D_18L_18R_A_18L_18R': 25, 'D_18C_18R_A_18C_18R': 26, 'D_18C_18L_18R_27_A_18L_18R_27': 27, 'D_27_36C_A_27_36R': 28, 'D_36L_36R_A_36L_36R': 29, 'D_36C_36L_36R_A_36C_36L_36R': 30},
             'kmia': {'D_12_8L_8R_9_A_12_8L_8R_9': 0, 'D_8L_8R_9_A_8L_8R_9': 1, 'D_26L_26R_27_30_A_26L_26R_27_30': 2, 'D_12_8L_8R_9_A_12_8R_9': 3, 'D_8R_9_A_8R_9': 4, 'other': 5, 'D_26L_26R_27_30_A_26L_27_30': 6, 'D_26L_27_A_26L_27': 7, 'D_8L_8R_A_8L_8R': 8, 'D_26L_26R_27_A_26L_26R_27': 9, 'D_8L_9_A_8L_9': 10, 'D_26L_26R_A_26L_26R': 11, 'D_26L_26R_27_A_26L_26R_27_30': 12, 'D_8L_8R_9_A_12_8L_8R_9': 13, 'D_8L_8R_A_8R': 14, 'D_26L_26R_27_A_26L_26R': 15, 'D_8L_8R_9_A_8R_9': 16, 'D_12_8R_9_A_12_8R_9': 17, 'D_12_8L_8R_A_12_8L_8R': 18, 'D_8L_9_A_9': 19, 'D_12_9_A_12_9': 20, 'D_26L_26R_27_A_26L_27_30': 21, 'D_26L_26R_27_A_26L_27': 22, 'D_26L_27_A_26L': 23, 'D_12_8L_8R_9_A_12_9': 24, 'D_12_8L_8R_9_A_12_8R': 25, 'D_26L_26R_A_26L': 26, 'D_12_8L_8R_9_A_8L_8R_9': 27},
             'kord': {'D_22L_28R_A_27C_27R_28C': 0, 'other': 1, 'D_10L_9C_A_10C_10R_9L': 2, 'D_22L_28R_A_28C': 3, 'D_22L_28R_A_27L_27R_28C': 4, 'D_28R_A_28C': 5, 'D_10L_A_10C': 6, 'D_22L_28R_A_27R_28C': 7, 'D_28C_A_28C': 8, 'D_10L_9C_A_10C': 9, 'D_28R_A_28R': 10, 'D_10L_9C_A_10C_9L': 11, 'D_10C_A_10C': 12, 'D_10L_A_10L': 13, 'D_10L_9R_A_10C': 14, 'D_10L_9R_A_10C_10R_9L': 15, 'D_9C_A_10C': 16, 'D_10L_9R_A_10C_9L': 17, 'D_22L_28R_A_28R': 18, 'D_27C_A_27C': 19, 'D_9C_A_9C': 20, 'D_10L_4L_A_10C': 21, 'D_22L_28R_A_27C': 22, 'D_22L_28R_A_27C_28C': 23, 'D_28R_A_27C': 24, 'D_10L_A_10C_10R_9L': 25, 'D_22L_28R_A_27L_28C': 26, 'D_10L_A_9C': 27, 'D_10L_22L_9C_A_10C_10R_9L': 28, 'D_10L_9R_A_10C_9L_9R': 29, 'D_10L_9C_A_10C_10R': 30, 'D_22L_28C_A_28C': 31, 'D_22L_A_22R': 32, 'D_4L_A_4R': 33, 'D_22L_28R_A_27C_27R': 34, 'D_28C_A_27L': 35, 'D_27C_28R_A_28R': 36, 'D_28R_A_27L_28R': 37},
             'kphx': {'D_25R_A_25L_26': 0, 'D_7L_A_7R_8': 1, 'D_7R_8_A_7R_8': 2, 'other': 3, 'D_7L_7R_A_7L_7R': 4, 'D_7L_8_A_7L_8': 5, 'D_25L_26_A_25L_26': 6, 'D_25R_A_25L_25R': 7, 'D_25R_26_A_25R_26': 8, 'D_7L_A_7L_7R': 9, 'D_7L_A_7R': 10, 'D_7L_7R_8_A_7L_7R_8': 11, 'D_25L_25R_A_25L_25R': 12, 'D_25R_A_25R': 13, 'D_7L_A_7L': 14, 'D_7L_8_A_7R_8': 15, 'D_25R_A_25R_26': 16, 'D_25R_26_A_25L_26': 17},
             'ksea': {'D_16L_A_16L_16R': 0, 'D_34R_A_34L_34R': 1, 'D_16L_A_16C_16L': 2, 'D_34R_A_34C_34R': 3, 'other': 4, 'D_16L_A_16R': 5, 'D_16C_A_16L_16R': 6, 'D_16L_A_16C_16R': 7, 'D_16C_A_16C_16R': 8, 'D_16L_A_16C': 9, 'D_34C_A_34C_34L': 10, 'D_34R_A_34C': 11}}


CONFIG_SUPPORT_DEFAULTS = {'katl': {0: 1.0,
  1: 0.939,
  2: 0.885,
  3: 0.838,
  4: 0.795,
  5: 0.756,
  6: 0.721,
  7: 0.687,
  8: 0.655,
  9: 0.626,
  10: 0.597,
  11: 0.572,
  12: 0.549},
 'kclt': {0: 1.0,
  1: 0.931,
  2: 0.874,
  3: 0.822,
  4: 0.774,
  5: 0.727,
  6: 0.683,
  7: 0.64,
  8: 0.599,
  9: 0.56,
  10: 0.522,
  11: 0.485,
  12: 0.452},
 'kden': {0: 1.0,
  1: 0.797,
  2: 0.625,
  3: 0.522,
  4: 0.44,
  5: 0.377,
  6: 0.322,
  7: 0.286,
  8: 0.252,
  9: 0.222,
  10: 0.194,
  11: 0.175,
  12: 0.162},
 'kdfw': {0: 1.0,
  1: 0.91,
  2: 0.832,
  3: 0.766,
  4: 0.708,
  5: 0.657,
  6: 0.609,
  7: 0.565,
  8: 0.523,
  9: 0.483,
  10: 0.446,
  11: 0.411,
  12: 0.378},
 'kjfk': {0: 1.0,
  1: 0.954,
  2: 0.911,
  3: 0.872,
  4: 0.838,
  5: 0.807,
  6: 0.779,
  7: 0.752,
  8: 0.726,
  9: 0.703,
  10: 0.682,
  11: 0.662,
  12: 0.643},
 'kmem': {0: 1.0,
  1: 0.844,
  2: 0.706,
  3: 0.587,
  4: 0.484,
  5: 0.399,
  6: 0.327,
  7: 0.268,
  8: 0.221,
  9: 0.189,
  10: 0.163,
  11: 0.141,
  12: 0.122},
 'kmia': {0: 1.0,
  1: 0.929,
  2: 0.865,
  3: 0.807,
  4: 0.755,
  5: 0.707,
  6: 0.662,
  7: 0.618,
  8: 0.576,
  9: 0.536,
  10: 0.497,
  11: 0.459,
  12: 0.422},
 'kord': {0: 1.0,
  1: 0.786,
  2: 0.615,
  3: 0.504,
  4: 0.427,
  5: 0.376,
  6: 0.344,
  7: 0.324,
  8: 0.312,
  9: 0.306,
  10: 0.296,
  11: 0.277,
  12: 0.253},
 'kphx': {0: 1.0,
  1: 0.943,
  2: 0.889,
  3: 0.838,
  4: 0.79,
  5: 0.744,
  6: 0.701,
  7: 0.66,
  8: 0.62,
  9: 0.583,
  10: 0.547,
  11: 0.515,
  12: 0.485},
 'ksea': {0: 1.0,
  1: 0.964,
  2: 0.931,
  3: 0.905,
  4: 0.88,
  5: 0.857,
  6: 0.835,
  7: 0.814,
  8: 0.794,
  9: 0.776,
  10: 0.758,
  11: 0.743,
  12: 0.73}}


TRANSITIONAL_CONFIGS = {'katl': {'D_26L_27R_A_26R_27L_28': 0,
                          'D_8R_9L_A_10_8L_9R': 0,
                          'other': 1,
                          'D_26L_27R_28_A_26R_27L_28': 0,
                          'D_26L_27R_A_26L_27L_28': 1,
                          'D_8R_9R_A_10_8L_9R': 1,
                          'D_26L_28_A_26L_28': 1,
                          'D_26R_28_A_26R_28': 1,
                          'D_26L_27R_A_26R_27L': 1,
                          'D_26L_27L_A_26R_27L_28': 1,
                          'D_26L_27R_A_26L_27R_28': 1,
                          'D_26L_27R_A_26R_27R_28': 1,
                          'D_26R_27R_A_26R_27L_28': 1,
                          'D_26L_27R_A_27L_28': 1,
                          'D_10_8R_A_10_8R': 1,
                          'D_8R_9L_A_8L_9R': 1,
                          'D_26L_28_A_26R_28': 1,
                          'D_26L_28_A_26R_27L_28': 1,
                          'D_8R_9L_A_10_8R_9R': 1,
                          'D_10_8R_9L_A_10_8L_9R': 0,
                          'D_26L_27R_A_26R_28': 1,
                          'D_10_8L_A_10_8L': 1,
                          'D_8R_9L_A_10_9R': 1,
                          'D_8L_9L_A_10_8L_9R': 1,
                          'D_8R_9L_A_8R_9L': 1,
                          'D_26L_27R_A_26L_27R': 1,
                          'D_9L_A_9R': 0},
                        'kclt': {'D_36C_36R_A_36C_36L_36R': 0,
                          'D_18C_18L_A_18C_18L_18R': 0,
                          'D_36C_36R_A_36C_36R': 1,
                          'D_36R_A_36R': 1,
                          'D_18C_18L_A_18C_18L': 1,
                          'D_18L_A_18L': 1,
                          'D_36R_A_36L_36R': 1,
                          'other': 1,
                          'D_36C_A_36C': 1,
                          'D_18C_A_18C': 1,
                          'D_18L_A_18L_18R': 1,
                          'D_36C_A_36C_36L': 0,
                          'D_18C_A_18C_18R': 1},
                        'kden': {'other': 1,
                          'D_25_34L_8_A_34R_35L_35R': 1,
                          'D_17L_25_8_A_16L_16R_17R': 1,
                          'D_25_34L_A_34R_35L_35R': 1,
                          'D_25_34L_8_A_35L_35R': 1,
                          'D_17L_8_A_16R_17R': 1,
                          'D_17L_25_8_A_16R_17R': 1,
                          'D_17L_17R_8_A_16L_16R_17R': 1,
                          'D_17L_8_A_16L_16R_17R': 1,
                          'D_25_34L_A_26_35L_35R': 1,
                          'D_34L_A_35L_35R': 1,
                          'D_34L_A_34R_35L_35R': 1,
                          'D_17R_A_16L_16R_17R': 1,
                          'D_25_34L_A_35L_35R': 1,
                          'D_34L_34R_A_35L_35R': 1,
                          'D_34L_8_A_35L_35R_7': 1,
                          'D_34L_34R_A_34R_35L_35R': 1,
                          'D_17R_8_A_16L_16R_17R': 1,
                          'D_34L_8_A_34R_35L_35R': 1,
                          'D_17L_8_A_16R_17R_7': 1,
                          'D_17R_A_16R_17R': 1,
                          'D_17R_8_A_16R_17R': 1,
                          'D_17R_A_16L_17R': 1,
                          'D_34L_8_A_35L_35R': 1,
                          'D_25_34L_34R_A_34R_35L_35R': 1,
                          'D_17L_17R_8_A_16R_17R': 1,
                          'D_17R_A_17R': 1,
                          'D_17L_17R_25_8_A_16L_16R_17R': 1,
                          'D_25_8_A_16R_35L_35R': 1,
                          'D_17L_17R_A_16L_16R_17R': 1,
                          'D_34L_A_35L': 1,
                          'D_34R_A_35L': 1,
                          'D_25_34L_34R_8_A_34R_35L_35R': 1,
                          'D_34L_34R_8_A_34R_35L_35R': 1,
                          'D_17L_8_A_17R': 1,
                          'D_17L_17R_A_16R_17R': 1,
                          'D_17L_25_A_16L_16R_17R': 1,
                          'D_17R_8_A_16L_17R': 1,
                          'D_34R_A_34R_35L_35R': 1,
                          'D_25_34L_34R_A_26_35L_35R': 1,
                          'D_17R_8_A_17R': 1,
                          'D_17L_17R_8_A_17R': 1},
                        'kdfw': {'D_17R_18L_A_13R_17C_17L_18R': 0,
                          'other': 1,
                          'D_17R_18L_A_13R_17C_17L_18L': 0,
                          'D_35L_36R_A_31R_35C_35R_36L': 0,
                          'D_17R_18L_A_13R_17C_17L': 0,
                          'D_35L_36R_A_31R_35C_35R': 0,
                          'D_17R_18L_A_17C_17R_18L': 1,
                          'D_17R_18L_A_17C_17R_18L_18R': 1,
                          'D_31L_35L_36R_A_31R_35C_35R_36R': 0,
                          'D_35L_36R_A_31R_35C_35R_36R': 1,
                          'D_35L_36R_A_35C_35L_36R': 1,
                          'D_17R_18L_A_17R_18L': 1,
                          'D_17C_18L_A_17C_18L': 1,
                          'D_17R_18L_A_17C_18L': 1,
                          'D_35L_36R_A_35C_36R': 1,
                          'D_17R_18L_A_17C_18R': 1,
                          'D_35L_36R_A_35C_35L_36L_36R': 1,
                          'D_35L_36R_A_35C_35R_36L': 0,
                          'D_17R_18L_A_17C_17L_18R': 1,
                          'D_35L_A_35C': 1,
                          'D_31L_35L_36R_A_31R_35C_35R': 1,
                          'D_35L_36R_A_35C_35R_36R': 0,
                          'D_17R_18L_A_17C_17L_18L': 1,
                          'D_35C_36R_A_35C_36R': 1,
                          'D_17R_A_17C_17R': 1,
                          'D_17C_A_17C': 1,
                          'D_17R_A_17C': 1,
                          'D_35L_A_35L': 1,
                          'D_18L_A_18L': 1,
                          'D_17R_18L_A_17R_18L_18R': 1,
                          'D_31L_35L_A_31R_35C_35R_36R': 0},
                        'kjfk': {'D_31L_A_31L_31R': 0,
                          'D_22R_A_22L_22R': 0,
                          'D_4L_A_4L_4R': 0,
                          'D_13R_A_13L_22L': 0,
                          'D_13R_A_13L': 1,
                          'D_22R_A_22L': 1,
                          'other': 1,
                          'D_31L_A_31L': 1,
                          'D_4L_A_4L': 1,
                          'D_22R_31L_A_22L_22R': 1,
                          'D_31L_4L_A_4L_4R': 1,
                          'D_31L_A_31R': 1,
                          'D_31L_31R_A_31L_31R': 0,
                          'D_13R_A_22L': 1},
                        'kmem': {'D_27_36C_36L_36R_A_27_36L_36R': 1,
                          'D_18C_18L_18R_A_18L_18R_27': 1,
                          'D_18C_18L_18R_A_18C': 1,
                          'other': 1,
                          'D_18C_18L_18R_A_18L_18R': 1,
                          'D_18C_18L_18R_27_A_27': 1,
                          'D_36C_36L_36R_A_36C': 1,
                          'D_36C_36L_36R_A_36L_36R': 1,
                          'D_36C_A_27_36L_36R': 1,
                          'D_18C_18L_18R_A_18C_18L_18R_27': 1,
                          'D_27_36C_36L_36R_A_27_36C_36L_36R': 1,
                          'D_36C_A_36L_36R_9': 1,
                          'D_18C_A_18L_18R_27': 1,
                          'D_36C_36L_36R_9_A_36L_36R_9': 1,
                          'D_18C_18L_18R_27_A_18C_18L_18R_27': 1,
                          'D_18C_A_18L_18R': 1,
                          'D_36C_A_36L_36R': 1,
                          'D_27_36L_36R_A_27_36L_36R': 1,
                          'D_27_36C_36L_A_27_36C_36L': 1,
                          'D_18C_18R_A_18C_18R_27': 1,
                          'D_18C_18L_A_18L_27': 1,
                          'D_18L_18R_A_18L_18R_27': 1,
                          'D_18C_18L_18R_A_18C_18L_18R': 1,
                          'D_36C_36L_A_36C_36L': 1,
                          'D_36C_36L_36R_A_27_36L_36R': 1,
                          'D_18L_18R_A_18L_18R': 1,
                          'D_18C_18R_A_18C_18R': 1,
                          'D_18C_18L_18R_27_A_18L_18R_27': 1,
                          'D_27_36C_A_27_36R': 1,
                          'D_36L_36R_A_36L_36R': 1,
                          'D_36C_36L_36R_A_36C_36L_36R': 1},
                        'kmia': {'D_12_8L_8R_9_A_12_8L_8R_9': 0,
                          'D_8L_8R_9_A_8L_8R_9': 0,
                          'D_26L_26R_27_30_A_26L_26R_27_30': 0,
                          'D_12_8L_8R_9_A_12_8R_9': 0,
                          'D_8R_9_A_8R_9': 1,
                          'other': 1,
                          'D_26L_26R_27_30_A_26L_27_30': 1,
                          'D_26L_27_A_26L_27': 1,
                          'D_8L_8R_A_8L_8R': 1,
                          'D_26L_26R_27_A_26L_26R_27': 1,
                          'D_8L_9_A_8L_9': 1,
                          'D_26L_26R_A_26L_26R': 1,
                          'D_26L_26R_27_A_26L_26R_27_30': 1,
                          'D_8L_8R_9_A_12_8L_8R_9': 0,
                          'D_8L_8R_A_8R': 1,
                          'D_26L_26R_27_A_26L_26R': 1,
                          'D_8L_8R_9_A_8R_9': 0,
                          'D_12_8R_9_A_12_8R_9': 1,
                          'D_12_8L_8R_A_12_8L_8R': 1,
                          'D_8L_9_A_9': 1,
                          'D_12_9_A_12_9': 1,
                          'D_26L_26R_27_A_26L_27_30': 1,
                          'D_26L_26R_27_A_26L_27': 1,
                          'D_26L_27_A_26L': 1,
                          'D_12_8L_8R_9_A_12_9': 0,
                          'D_12_8L_8R_9_A_12_8R': 1,
                          'D_26L_26R_A_26L': 0,
                          'D_12_8L_8R_9_A_8L_8R_9': 1},
                        'kord': {'D_22L_28R_A_27C_27R_28C': 0,
                          'other': 1,
                          'D_10L_9C_A_10C_10R_9L': 0,
                          'D_22L_28R_A_28C': 1,
                          'D_22L_28R_A_27L_27R_28C': 1,
                          'D_28R_A_28C': 1,
                          'D_10L_A_10C': 1,
                          'D_22L_28R_A_27R_28C': 1,
                          'D_28C_A_28C': 1,
                          'D_10L_9C_A_10C': 1,
                          'D_28R_A_28R': 1,
                          'D_10L_9C_A_10C_9L': 1,
                          'D_10C_A_10C': 1,
                          'D_10L_A_10L': 1,
                          'D_10L_9R_A_10C': 1,
                          'D_10L_9R_A_10C_10R_9L': 1,
                          'D_9C_A_10C': 1,
                          'D_10L_9R_A_10C_9L': 1,
                          'D_22L_28R_A_28R': 1,
                          'D_27C_A_27C': 1,
                          'D_9C_A_9C': 1,
                          'D_10L_4L_A_10C': 1,
                          'D_22L_28R_A_27C': 1,
                          'D_22L_28R_A_27C_28C': 1,
                          'D_28R_A_27C': 1,
                          'D_10L_A_10C_10R_9L': 1,
                          'D_22L_28R_A_27L_28C': 1,
                          'D_10L_A_9C': 0,
                          'D_10L_22L_9C_A_10C_10R_9L': 1,
                          'D_10L_9R_A_10C_9L_9R': 1,
                          'D_10L_9C_A_10C_10R': 1,
                          'D_22L_28C_A_28C': 1,
                          'D_22L_A_22R': 1,
                          'D_4L_A_4R': 1,
                          'D_22L_28R_A_27C_27R': 1,
                          'D_28C_A_27L': 1,
                          'D_27C_28R_A_28R': 1,
                          'D_28R_A_27L_28R': 1},
                        'kphx': {'D_25R_A_25L_26': 0,
                          'D_7L_A_7R_8': 0,
                          'D_7R_8_A_7R_8': 1,
                          'other': 0,
                          'D_7L_7R_A_7L_7R': 1,
                          'D_7L_8_A_7L_8': 1,
                          'D_25L_26_A_25L_26': 1,
                          'D_25R_A_25L_25R': 1,
                          'D_25R_26_A_25R_26': 1,
                          'D_7L_A_7L_7R': 1,
                          'D_7L_A_7R': 1,
                          'D_7L_7R_8_A_7L_7R_8': 1,
                          'D_25L_25R_A_25L_25R': 1,
                          'D_25R_A_25R': 1,
                          'D_7L_A_7L': 1,
                          'D_7L_8_A_7R_8': 1,
                          'D_25R_A_25R_26': 1,
                          'D_25R_26_A_25L_26': 1},
                        'ksea': {'D_16L_A_16L_16R': 0,
                          'D_34R_A_34L_34R': 0,
                          'D_16L_A_16C_16L': 1,
                          'D_34R_A_34C_34R': 1,
                          'other': 1,
                          'D_16L_A_16R': 0,
                          'D_16C_A_16L_16R': 1,
                          'D_16L_A_16C_16R': 1,
                          'D_16C_A_16C_16R': 1,
                          'D_16L_A_16C': 1,
                          'D_34C_A_34C_34L': 1,
                          'D_34R_A_34C': 1}}

  
# MIN_SUPPORTS = [
#     1e-7,
#     1e-5,
#     1e-4,
#     1e-3,
#     1e-2,
#     0.05
# ]

MIN_SUPPORTS = [
    0.001
]


GUFI_PARTS = ['gufi_flight_number', 'gufi_origin','gufi_destination','gufi_date','gufi_sch_dep','gufi_sch_arr','gufi_tracking']
