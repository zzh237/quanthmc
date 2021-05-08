import torch 
import numpy as np 
from data.classification_data import * 
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, datasets
# from keras.datasets import mnist

class mnist_data(classification_data):
    def __init__(self, args):
        self.args = args  
        if self.args.data_name == 'mnist':
            self.args.target_class = 10
        else:
            self.args.target_class = 1 

    def load_data(self):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
        trainset = datasets.MNIST(root='././data/datasets',
                                               train=True,
                                               transform=transform_train,
                                               download=True)
        valset = datasets.MNIST(root='././data/datasets', train=False, download=True, transform=transform_test)
        
        
        return (trainset, valset) 
    
    def __repr__(self):
        if self.args.data_name == 'mnist':
            return 'mnist'
        else:
            return "mnist_two_target"

        
    def train_test_data(self)->dict:
        if self.args.data_name == 'mnist':
            data = self.load_data() 
        if self.args.data_name == 'mnist_two_target':
            data = self.load_binary_with_padding(0, 1)   
        return data 

    
    def data_loading_with_target(self, target_num):
        # Should clean out the test data set also!!!!!!

        (train, test) = self.load_data()
        x_train, y_train = train.train_data, train.train_labels
        x_test, y_test = test.test_data, test.test_labels



        (x_train, y_train), (x_test, y_test) = self.load_data()

        # Normalized

        x_train = (x_train.reshape(60000, 784)/255. - 0.1307)/0.3081
        x_test = (x_test.reshape(10000, 784)/255. - 0.1307)/0.3081

        # Select out the train target

        x_true_train = []
        y_true_train = []

        for idx in range(len(y_train)):
            if y_train[idx] == target_num:
                x_true_train.append(x_train[idx])
                y_true_train.append(y_train[idx])

        x_true_train = np.array(x_true_train)
        y_true_train = np.array(y_true_train)

        # Select out the test target

        x_true_test = []
        y_true_test = []

        for idx in range(len(y_test)):
            if y_test[idx] == target_num:
                x_true_test.append(x_test[idx])
                y_true_test.append(y_test[idx])

        x_true_train = np.array(x_true_train)
        y_true_train = np.array(y_true_train)

        x_true_test = np.array(x_true_test)
        y_true_test = np.array(y_true_test)

        # x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
        x_train = torch.from_numpy(x_true_train).type(torch.FloatTensor)
        x_test = torch.from_numpy(x_true_test).type(torch.FloatTensor)

        # y_train = torch.from_numpy(y_train).type(torch.LongTensor)
        y_train = torch.from_numpy(y_true_train).type(torch.LongTensor)
        y_test = torch.from_numpy(y_true_test).type(torch.LongTensor)

        return x_train, y_train, x_test, y_test

    def data_loading_with_padding_target(self, target_num):
        (train, test) = self.load_data()
        x_train, y_train = train.train_data, train.train_labels
        x_test, y_test = test.test_data, test.test_labels

        x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant')
        x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant')

        # Normalized

        x_train = (x_train.reshape(60000, 1024)/255. - 0.1307)/0.3081
        x_test = (x_test.reshape(10000, 1024)/255. - 0.1307)/0.3081

        # Select out the train target

        x_true_train = []
        y_true_train = []

        for idx in range(len(y_train)):
            if y_train[idx] == target_num:
                x_true_train.append(x_train[idx])
                y_true_train.append(y_train[idx])

        x_true_train = np.array(x_true_train)
        y_true_train = np.array(y_true_train)

        # Select out the test target

        x_true_test = []
        y_true_test = []

        for idx in range(len(y_test)):
            if y_test[idx] == target_num:
                x_true_test.append(x_test[idx])
                y_true_test.append(y_test[idx])

        x_true_train = np.array(x_true_train)
        y_true_train = np.array(y_true_train)

        x_true_test = np.array(x_true_test)
        y_true_test = np.array(y_true_test)

        # x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
        x_train = torch.from_numpy(x_true_train).type(torch.FloatTensor)
        x_test = torch.from_numpy(x_true_test).type(torch.FloatTensor)

        # y_train = torch.from_numpy(y_train).type(torch.LongTensor)
        y_train = torch.from_numpy(y_true_train).type(torch.LongTensor)
        y_test = torch.from_numpy(y_true_test).type(torch.LongTensor)

        return x_train, y_train, x_test, y_test


    def data_loading_two_target(self, target_1, target_2, padding = False):
        # Define the first data be labeled 0
        # Define the second data be labeled 1
        x_train_first = None
        y_train_first = None
        x_test_first = None
        y_test_first = None
        x_train_second = None
        y_train_second = None
        x_test_second = None
        y_test_second = None

        if padding == True:
            x_train_first, y_train_first, x_test_first, y_test_first = self.data_loading_with_padding_target(target_1)
            x_train_second, y_train_second, x_test_second, y_test_second = self.data_loading_with_padding_target(target_2)

        else:
            x_train_first, y_train_first, x_test_first, y_test_first = self.data_loading_with_target(target_1)
            x_train_second, y_train_second, x_test_second, y_test_second = self.data_loading_with_target(target_2)

        x_train_combined = np.concatenate((x_train_first, x_train_second), axis = 0)
        y_train_combined = np.concatenate((y_train_first, y_train_second), axis = 0)

        x_test_combined = np.concatenate((x_test_first, x_test_second), axis = 0)
        y_test_combined = np.concatenate((y_test_first, y_test_second), axis = 0)

        length_first_train_set = len(x_train_first)
        length_second_train_set = len(x_train_second)

        length_first_test_set = len(x_test_first)
        length_second_test_set = len(x_test_second)

        permutation_train = np.random.permutation(length_first_train_set + length_second_train_set)
        permutation_test = np.random.permutation(length_first_test_set + length_second_test_set)

        x_train_combined = x_train_combined[permutation_train]
        y_train_combined = y_train_combined[permutation_train]

        x_test_combined = x_test_combined[permutation_test]
        y_test_combined = y_test_combined[permutation_test]

        # Transform the y label to 0 or 1
        y_train_combined = (y_train_combined == target_2)
        y_test_combined = (y_test_combined == target_2)

        # x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
        x_train = torch.from_numpy(x_train_combined).type(torch.FloatTensor)

        # y_train = torch.from_numpy(y_train).type(torch.LongTensor)
        y_train = torch.from_numpy(y_train_combined).type(torch.FloatTensor)

        x_test = torch.from_numpy(x_test_combined).type(torch.FloatTensor)

        # y_train = torch.from_numpy(y_train).type(torch.LongTensor)
        y_test = torch.from_numpy(y_test_combined).type(torch.FloatTensor)

        return x_train, y_train, x_test, y_test

    def load_binary_with_padding(self, target_num_1 = 6, target_num_2 = 8):

        x_train, y_train, x_test, y_test = self.data_loading_two_target(target_1 = target_num_1, target_2 = target_num_2, padding = True)
        

        y_train = np.reshape(y_train, (-1,1)) #reshape the y to have one more dimension to fit the requirement of torch output 
        y_test = np.reshape(y_test, (-1,1)) #reshape the y to have one more dimension to fit the requirement of torch output 
        np.random.seed(0)
        num_data = len(x_train)
        self.args.N_tr = int(self.args.tr_ratio * num_data)

        index = np.random.permutation(range(num_data))

        x_for_train = x_train[index[:self.args.N_tr]]
        y_for_train = y_train[index[:self.args.N_tr]]

        x_for_val = x_train[index[self.args.N_tr:]]
        y_for_val = y_train[index[self.args.N_tr:]]

        x_for_test = x_test
        y_for_test = y_test

        mnist_train= TensorDataset(x_for_train,y_for_train)
        # mnist_val= TensorDataset(x_for_val,y_for_val)
        mnist_test = TensorDataset(x_for_test,y_for_test)
        return (mnist_train,mnist_test) 

        # return x_for_train, y_for_train, x_for_val, y_for_val, x_for_test, y_for_test