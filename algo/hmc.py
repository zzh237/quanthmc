import algo.hamiltorch as hmt
import torch 
import torch.nn.functional as F
from algo.algo_interface import * 
from torch.utils.data import TensorDataset, DataLoader
import os

class hmc(algo_interface):
    def __init__(self, args, model, data):
        self.model = model
        self.data = data  
        self.args = args 
        

        # self.x_train, self.y_train = data['x_train'], data["y_train"]
        # self.x_val, self.y_val = data["x_val"], data["y_val"]
        if self.args.data_name == 'mnist':
            data_train, data_test = data[0], data[1]
            train_size = data_train.train_data.shape[0]
            val_size = data_test.test_data.shape[0]
            self.trainloader = DataLoader(data_train, batch_size=train_size, shuffle=True, pin_memory=False,
                                              num_workers=self.args.num_workers)
            self.valloader = DataLoader(data_test, batch_size=val_size, shuffle=False, pin_memory=False,
                                            num_workers=self.args.num_workers)


            self.x_train, self.y_train = data[0].train_data, data[0].train_labels
            self.x_val, self.y_val = data[1].train_data, data[1].train_labels

            self.x_train = self.x_train.view(self.x_train.shape[0], -1)
            self.x_val = self.x_val.view(self.x_val.shape[0], -1)

        else:
            self.x_train, self.y_train = data[0].tensors[0],data[0].tensors[1]
            self.x_val, self.y_val = data[1].tensors[0],data[1].tensors[1]
    
        tau_list = []
        tau = self.args.tau#/100. # iris 1/10
        for w in self.model.parameters():
            tau_list.append(self.args.tau)
        self.tau_list = torch.tensor(tau_list).to(self.args.device )

        self.losstype = "regression"

        if self.args.target_class >= 3:
            self.losstype = "multi_class_linear_output" #this will use CrossEntropy loss
        if self.args.target_class == 1:
            self.losstype = "binary_class_linear_output" # this will use BCEwithlogits loss
        
    def __repr__(self):
        return 'HMC'

    def fit(self):
        hmt.set_random_seed(123)
        params_init = hmt.util.flatten(self.model).to(self.args.device ).clone()
        self.params_hmc, result = hmt.sample_model(self.model, self.x_train, self.y_train, model_loss=self.losstype, params_init=params_init, num_samples=self.args.num_samples,
							   step_size=self.args.step_size, num_steps_per_sample=self.args.L,tau_out=self.args.tau_out,tau_list=self.tau_list)
        
        output_path = os.path.join(self.args.out_dir, 'hmc_result.npy')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, result)
    def predict(self):
        
        pred_list, log_prob_list = hmt.predict_model(self.model, x=self.x_val, y=self.y_val, samples=self.params_hmc[:], model_loss=self.losstype, tau_out=1., tau_list=self.tau_list)
        
        err = torch.zeros( len(pred_list)-1)
        nll = torch.zeros( len(pred_list)-1)
        
        if self.args.target_class == 1:
            
            pred = pred_list.detach().clone()
            pred_reduced = pred_list.detach().clone()
            
            pred = torch.squeeze(pred_list) 
            pred_reduced = torch.squeeze(pred_list) 

            pred[pred >= 0] = 1
            pred[pred < 0] = 0
            
            best_error = (pred.float() != self.y_val.flatten()).sum().float()/self.y_val.shape[0]
            
            ensemble_logits = pred_reduced[0]
            
            for s in range(1,len(pred_list)):

                single_pred = pred_reduced[s]
                single_pred[single_pred >= 0] = 1
                single_pred[single_pred < 0] = 0
                error = (single_pred.float() != self.y_val.flatten()).sum().float()/self.y_val.shape[0]
                if error < best_error:
                    best_error = error


                reduced_mean = pred_reduced[:s].mean(0)
                reduced_mean[reduced_mean >= 0] = 1
                reduced_mean[reduced_mean < 0] = 0
                
                
                
                err[s-1] = (reduced_mean.float() != self.y_val.flatten()).sum().float()/self.y_val.shape[0]
                ensemble_logits += pred_reduced[s]
                ensemble_logits = ensemble_logits.cpu()/(s+1)
                
                nll[s-1] = F.binary_cross_entropy_with_logits(ensemble_logits, self.y_val[:].cpu().flatten(), reduction='mean')

        else:
            _, pred = torch.max(pred_list, 2) #return value and index
        
            best_error = (pred.float() != self.y_val.flatten()).sum().float()/self.y_val.shape[0]

            ensemble_proba = F.softmax(pred_list[0], dim=-1)
        
            for s in range(1,len(pred_list)):
                _, pred = torch.max(pred_list[:s].mean(0), -1)
                _, single_pred = torch.max(pred_list[s], -1)

                error = (single_pred.float() != self.y_val.flatten()).sum().float()/self.y_val.shape[0]
                if error < best_error:
                    best_error = error
                
                err[s-1] = (pred.float() != self.y_val.flatten()).sum().float()/self.y_val.shape[0]
                ensemble_proba += F.softmax(pred_list[s], dim=-1)
                nll[s-1] = F.nll_loss(torch.log(ensemble_proba.cpu()/(s+1)), self.y_val[:].long().cpu().flatten(), reduction='mean')

        return err, best_error, nll











