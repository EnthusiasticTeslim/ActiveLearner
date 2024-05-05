import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd

# dataset preparation
class Data:
    def __init__(self, data: pd.DataFrame, 
                 features: list, target: str, random_state: int, test_size: float, handler):
        # extract data
        X = data[features].values
        y = data[target].values
        # scale the X data
        print('Scaling the data')
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        # split data
        print('Splitting the data')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        # handler
        self.handler = handler
        self.n_pool = len(self.X_train)
        self.n_test = len(self.X_test)
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        
    def initialize_labels(self, num):
        '''initialize the labels of the pool data'''
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_labeled_data(self):
        '''return the labeled data'''
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.y_train[labeled_idxs])
    
    def get_unlabeled_data(self):
        '''return the unlabeled data'''
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.y_train[unlabeled_idxs])
    
    def get_train_data(self):
        '''return the train data'''
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.y_train)
        
    def get_test_data(self):
        '''return the test data'''
        return self.handler(self.X_test, self.y_test)

# dataset handler
class Handler(Dataset):
    '''dataset handler to handle access to the data'''
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)
    
