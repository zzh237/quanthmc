
import torch.nn.functional as F
import numpy as np 
from algo.hamiltorch.util import * 
from algo.Stochastic_Gradient_HMC_SA.utils import *
import torch.nn as nn 
import copy

class adam_updater():
    def __init__(self, args, model):
        self.model = model
        self.args = args 
        # self.x_train, self.y_train = self.data['x_train'],self.data['y_train']
        # self.x_val, self.y_val = self.data['x_val'],self.data['y_val']
    
        self.lr = self.args.lr
        # self.model = MLP(input_dim=784, width=1200, depth=2, output_dim=10)
        if self.args.device == 'cuda':
            self.cuda = True
        else:
            self.cuda = False

        self.N_train = self.args.N_tr
        # self.create_net()
        self.create_opt()
      

        self.grad_buff = []
        self.max_grad = 1e20
        self.grad_std_mul = self.args.grad_std_mul

        self.weight_set_samples = []

    

    def create_opt(self):
        """This optimiser incorporates the gaussian prior term automatically. The prior variance is gibbs sampled from
        its posterior using a gamma hyper-prior."""
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def onestep(self, x, y):
        self.model.train()
        # x, y = to_variable(var=(x, y.long()), cuda=self.cuda)
        self.optimizer.zero_grad() 
        # print("### x is on GPU", x.is_cuda)
        # print("### y is on GPU", y.is_cuda) 
        print("### model is on GPU", next(self.model.parameters()).is_cuda) 
        
        out = self.model(x)
        if self.args.target_class ==1: #binary classification, the out is still logits
            # print("#### x is on cuda", x.is_cuda)
            # print("#### out is on cuda", out.is_cuda)
            # print("#### y is on cuda", y.is_cuda)
            
            loss = F.binary_cross_entropy_with_logits(out, y, reduction='mean')
        if self.args.target_class >2: #multiclassiciation , y is in 1D, and label is 0,1,2,3,4
            loss = F.cross_entropy(out, y, reduction='mean') 
        
        loss = loss * self.N_train  # We use mean because we treat as an estimation of whole dataset
        loss.backward()

        # Gradient buffer to allow for dynamic clipping and prevent explosions
        if len(self.grad_buff) > 1000:
            self.max_grad = np.mean(self.grad_buff) + self.grad_std_mul * np.std(self.grad_buff)
            self.grad_buff.pop(0)

#             File "<__array_function__ internals>", line 6, in mean
# 15579
#   File "/usr/local/lib/python3.6/dist-packages/numpy/core/fromnumeric.py", line 3373, in mean
# 15580
#     out=out, **kwargs)
# 15581
#   File "/usr/local/lib/python3.6/dist-packages/numpy/core/_methods.py", line 170, in _mean
# 15582
#     ret = ret.dtype.type(ret / rcount)
# 15583
# AttributeError: 'torch.dtype' object has no attribute 'type'    


        # Clipping to prevent explosions
        self.grad_buff.append(nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                                       max_norm=self.max_grad, norm_type=2))
        if self.grad_buff[-1] >= self.max_grad:
            print(self.max_grad, self.grad_buff[-1])
            self.grad_buff.pop()
        self.optimizer.step()

        if self.args.target_class ==1:
            pred = out.detach().clone() 
            pred = torch.where(pred>=0, 1, 0)
        if self.args.target_class >2:    
        # out: (batch_size, out_channels, out_caps_dims)
            pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        
        err = pred.ne(y.data).sum()

        return loss.data * x.shape[0] / self.N_train, err

    def save_sampled_net(self, max_samples):

        if len(self.weight_set_samples) >= max_samples:
            self.weight_set_samples.pop(0)

        self.weight_set_samples.append(copy.deepcopy(self.model.state_dict()))

        cprint('c', ' saving weight samples %d/%d' % (len(self.weight_set_samples), max_samples))
        return None
    
    def eval(self, x, y, train=False):
        self.model.eval()
        # x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out = self.model(x)

        if self.args.target_class ==1: #binary classification, the out is still logits
            loss = F.binary_cross_entropy_with_logits(out, y, reduction='sum')
            pred = out.detach().clone() 
            pred = torch.where(pred>=0, 1, 0)

            probs = torch.sigmoid(out)
        
        if self.args.target_class >2: #multiclassiciation , y is in 1D, and label is 0,1,2,3,4
            loss = F.cross_entropy(out, y, reduction='sum')
            probs = F.softmax(out, dim=1).data.cpu()
            pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def sample_predict(self, x, Nsamples=0, grad=False):
        """return predictions using multiple samples from posterior"""
        self.model.eval()
        if Nsamples == 0:
            Nsamples = len(self.weight_set_samples)
        x, = to_variable(var=(x, ), cuda=self.cuda)

        if grad:
            self.optimizer.zero_grad()
            if not x.requires_grad:
                x.requires_grad = True

        out = x.data.new(Nsamples, x.shape[0], self.model.output_dim)

        # iterate over all saved weight configuration samples
        for idx, weight_dict in enumerate(self.weight_set_samples):
            if idx == Nsamples:
                break
            self.model.load_state_dict(weight_dict)
            out[idx] = self.model(x)

        out = out[:idx]
        prob_out = F.softmax(out, dim=2)

        if grad:
            return prob_out
        else:
            return prob_out.data

    def get_weight_samples(self, Nsamples=0):
        """return weight samples from posterior in a single-column array"""
        weight_vec = []

        if Nsamples == 0 or Nsamples > len(self.weight_set_samples):
            Nsamples = len(self.weight_set_samples)

        for idx, state_dict in enumerate(self.weight_set_samples):
            if idx == Nsamples:
                break

            for key in state_dict.keys():
                if 'weight' in key:
                    weight_mtx = state_dict[key].cpu().data
                    for weight in weight_mtx.view(-1):
                        weight_vec.append(weight)

        return np.array(weight_vec)

    def save_weights(self, filename):
        save_object(self.weight_set_samples, filename)

    def load_weights(self, filename, subsample=1):
        self.weight_set_samples = load_object(filename)
        self.weight_set_samples = self.weight_set_samples[::subsample]