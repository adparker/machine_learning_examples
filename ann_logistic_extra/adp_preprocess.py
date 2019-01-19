import numpy as np
import pandas as pd
df = pd.read_csv('ecommerce_data.csv')

def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.values
    
    # split out the last column
    X = data[:, :-1]
    Y = data[:, -1]
    
    # normalize the numerical columns (n_products_viewed, visit_duration)
    X[:,1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X[:,2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()
    
    # one hot encode categorical values 
    # time_of_day - column index 4
    N, D = X.shape
    X2 = np.zeros((N, D+3))
    # copy unchanged columns
    X2[:, 0:(D-1)] = X[:, 0:(D-1)] 
    X2[np.arange(N), (D-1) + X[:, D-1].astype(np.int32) ] = 1
    
    return X2, Y

def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2


