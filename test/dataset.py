import pandas as pd
import os
import numpy as np
import ast

def dataset_transform_test():

    '''
    Load data set
    '''
    # pull train data
    data_path = os.getcwd() + '/data'
    for i in os.listdir(data_path):
        if i == 'train.csv':
            df = pd.read_csv(data_path + '/' + i)

    '''
    Feature creation / transformation
    '''

    # attach prompt to each response
    for i in ['response_a','response_b']:
        df[i] = "PROMPT: " + df['prompt'] + " RESPONSE: " + df[i]

    # give ties as indiviual winners
    for i in ['a','b']:
        df[f'winner_model_{i}'] = np.where(df['winner_tie'] == 1, 1, df[f'winner_model_{i}'])

    '''
    Dataset Transformation, wide to long    
    '''

    # wide to long transformation
    data = pd.melt(
        df, 
        id_vars=['id','prompt'], 
        value_vars=['response_a', 'response_b'],
        var_name='response_type', 
        value_name='response'
    ).reset_index(drop=True) # Reset index so it matches the new row count

    # adjust features
    for i in ['winner_model','model']:
        data[i] = pd.melt(
            df, 
            id_vars=['prompt'], 
            value_vars=[f'{i}_a',f'{i}_b'],
            var_name= f'{i}_type', 
            value_name= i
        ).reset_index(drop=True)[i]

    # subset and rename
    data = data['id response winner_model model'.split()].rename(columns = {'response':'text','winner_model':'results'})

    # add model features
    data = data.merge(pd.read_csv(data_path + '/_feature_model_categories.csv')['model model_family submodel version'.split()], how = 'left', on = 'model')

    print(data)
if __name__ == "__main__":

    os.chdir('..')


    dataset_transform_test()