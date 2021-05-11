
import torch 
import numpy as np 
from data.classification_data import * 
from sklearn.datasets import load_breast_cancer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

class cancer_data(classification_data):
    def __init__(self, args):
        self.args = args  

    def load_data(self):
        data = load_breast_cancer()
        return data 
    def __repr__(self):
        return 'cancer'

        
    def train_test_data(self)->dict:
        data = self.load_data()
        x_ = data['data']
        if self.args.model_name == 'Quant':
            x_ = np.pad(x_, ((0,0),(0,2)), 'constant')
        y_ = data['target']
        
        y_ = np.reshape(y_, (-1,1)) #reshape the y to have one more dimension to fit the requirement of torch output 
        a = np.arange(x_.shape[0])
        self.args.N_tr = int(np.floor(x_.shape[0] * self.args.tr_ratio)) 
        self.args.feature_dim = x_.shape[1]
      
        self.args.target_class = 1
        
        train_index = np.random.choice(a, size = self.args.N_tr, replace = False)
        val_index = np.delete(a, train_index, axis=0)
        x_train = x_[train_index]
        y_train = y_[train_index]
        x_val = x_[val_index][:]
        y_val = y_[val_index][:]

        
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        
        x_train = torch.FloatTensor(x_train)
        y_train = torch.FloatTensor(y_train)
        x_val = torch.FloatTensor(x_val)
        y_val = torch.FloatTensor(y_val)
        
        # plt.scatter(x_train.numpy()[:,0],y_train.numpy())
            
        x_train = x_train.to(self.args.device)
        y_train = y_train.to(self.args.device)
        x_val = x_val.to(self.args.device)
        y_val = y_val.to(self.args.device)
        data = {'x_train':x_train, "y_train":y_train, "x_val":x_val, "y_val":y_val}

        cancer_train= TensorDataset(data['x_train'],data['y_train'])
        cancer_test= TensorDataset(data['x_val'],data['y_val'])
        return (cancer_train,cancer_test) 