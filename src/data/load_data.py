import pandas as pd
import os

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    train_file = os.path.join(current_dir, "../../data/raw/air_system_previous_years.csv")
    val_file = os.path.join(current_dir, "../../data/raw/air_system_present_year.csv")
    
    train = pd.read_csv(train_file)
    val = pd.read_csv(val_file)

    print("--------------------------------------------------------------------------------")
    print("Data loaded.")
    print("--------------------------------------------------------------------------------")
    
    return train, val