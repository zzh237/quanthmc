import torch 
import torch.nn.functional as F
import numpy as np 
from algo.algo_interface import * 
from torch.utils.data import DataLoader
from algo.Stochastic_Gradient_HMC_SA.utils import * 
from algo.gradient_based.adam_update import * 
import time 

class adam(algo_interface):
    def __init__(self, args, model, data):
        self.model = model
        self.data = data  
        if args.data_name == 'mnist':
            args.flat_ims = True 
        
        data_train, data_test = data 
        

        self.trainloader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, pin_memory=False,
                                              num_workers=args.num_workers)
        self.valloader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, pin_memory=False,
                                            num_workers=args.num_workers)


        self.args = args 
        self.schedule = None
        self.updater = adam_updater(args, model)
        self.lr = self.args.lr
        self.epoch = 0

        
    def __repr__(self):
        return 'Adam'
    
    def predict(self):

        
        # net dims
        epoch = 0
        it_count = 0
        ## ---------------------------------------------------------------------------------------------------------------------
        # train
        cprint('c', '\nTrain:')

        print('  init cost variables:')
        cost_train = np.zeros(self.args.epochs)
        err_train = np.zeros(self.args.epochs)
        cost_dev = np.zeros(self.args.epochs)
        err_dev = np.zeros(self.args.epochs)
        best_cost = np.inf
        best_err = np.inf

        tic0 = time.time()
        for i in range(epoch, self.args.epochs):
            self.set_mode_train(True)
            tic = time.time()
            nb_samples = 0
            for x, y in self.trainloader:

                if self.args.flat_ims:
                    x = x.view(x.shape[0], -1)

                cost_pred, err = self.updater.onestep(x, y)
                print('##batch finished', err)
                
                it_count += 1
                err_train[i] += err
                cost_train[i] += cost_pred
                nb_samples += len(x)

            cost_train[i] /= nb_samples
            err_train[i] /= nb_samples
            toc = time.time()

            # ---- print
            print("it %d/%d, Jtr_pred = %f, err = %f, " % (i, self.args.epochs, cost_train[i], err_train[i]), end="")
            cprint('r', '   time: %f seconds\n' % (toc - tic))
            self.update_lr(i)

            
            # ---- dev
            if i % self.args.log_interval == 0:
                nb_samples = 0
                for j, (x, y) in enumerate(self.valloader):
                    if self.args.flat_ims:
                        x = x.view(x.shape[0], -1)

                    cost, err, probs = self.updater.eval(x, y)

                    cost_dev[i] += cost
                    err_dev[i] += err
                    nb_samples += len(x)

                cost_dev[i] /= nb_samples
                err_dev[i] /= nb_samples

                cprint('g', '    Jdev = %f, err = %f\n' % (cost_dev[i], err_dev[i]))
                if err_dev[i] < best_err:
                    best_err = err_dev[i]
                    cprint('b', 'best test error')

        toc0 = time.time()
        runtime_per_it = (toc0 - tic0) / float(self.args.epochs)
        cprint('r', '   average time: %f seconds\n' % runtime_per_it)
        return err_dev, np.min(err_dev), cost_dev

    
    def update_lr(self, epoch, gamma=0.99):
        self.epoch += 1
        if self.schedule is not None:
            if len(self.schedule) == 0 or epoch in self.schedule:
                self.lr *= gamma
                print('learning rate: %f  (%d)\n' % self.lr, epoch)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

    
    def get_nb_parameters(self):
        return np.sum(p.numel() for p in self.model.parameters())

    def set_mode_train(self, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def update_lr(self, epoch, gamma=0.99):
        self.epoch += 1
        if self.schedule is not None:
            if len(self.schedule) == 0 or epoch in self.schedule:
                self.lr *= gamma
                print('learning rate: %f  (%d)\n' % self.lr, epoch)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

    def save(self, filename):
        cprint('c', 'Writting %s\n' % filename)
        torch.save({
            'epoch': self.epoch,
            'lr': self.lr,
            'model': self.model,
            'optimizer': self.optimizer}, filename)

    def load(self, filename):
        cprint('c', 'Reading %s\n' % filename)
        state_dict = torch.load(filename)
        self.epoch = state_dict['epoch']
        self.lr = state_dict['lr']
        self.model = state_dict['model']
        self.optimizer = state_dict['optimizer']
        print('  restoring epoch: %d, lr: %f' % (self.epoch, self.lr))
        return self.epoch