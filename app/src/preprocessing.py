# Import libraries
import pandas as pd
import numpy as np

# Define column types
target_col = 'binary_target'
colums_used = ['секретный_скор', 'on_net', 'доход']
drop_columns = ['client_id']

def import_data(path_to_file):

    # Get input dataframe
    input_df = pd.read_csv(path_to_file).drop(columns=drop_columns)
    return input_df

# Main preprocessing function
def run_preproc(input_df):
    # Create dataframe 
    input_df["регион"].fillna("NaN", inplace=True)
    input_df["pack"].fillna("NaN", inplace=True)    
    output_df = input_df

    # Return resulting dataset
    return output_df