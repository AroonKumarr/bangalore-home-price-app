import pickle
import json
import numpy as np
import os

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    # Automatically get path to artifacts regardless of current working directory
    base_dir = os.path.dirname(__file__)
    artifacts_path = os.path.join(base_dir, "artifacts")

    # Load columns.json
    columns_file = os.path.join(artifacts_path, "columns.json")
    with open(columns_file, "r") as f:
        __data_columns = json.load(f)["data_columns"]
        __locations = __data_columns[3:]

    # Load the model
    model_file = os.path.join(artifacts_path, "banglore_home_prices_model.pickle")
    with open(model_file, "rb") as f:
        __model = pickle.load(f)

    print("loading saved artifacts...done ✅")
    print("Model loaded:", type(__model))


def get_location_names():
    return __locations


def get_data_columns():
    return __data_columns
