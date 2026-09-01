'''
Understand summary statistics of the meta data
'''
import pandas as pd
import numpy as np
import os 

class Metadata:

    def __init__(self, path):

        self.df_train = pd.read_csv(path + '/train.csv')

        self.df_train_series = pd.read_csv(path + '/train_series.csv')

    def _missing_train_data(self, clm):

        numerator = len(self.df_train[self.df_train[clm].isna() == True].index)

        demoninator = len(self.df.index)

        print(numerator)
        return np.round(numerator/demoninator,3) * 100


    def __call__(self, *args, **kwds):

        print('From train.csv ...')
        [print(f'{self._missing_train_data(clm)}% of missing data from {clm}') for clm in self.df_train.columns.to_list()]

        print('Images within each train series')

        pass

if __name__ == '__main__':

    os.chdir('..')

    df = pd.read_csv(os.getcwd() + '/data')

    Metadata(df)()