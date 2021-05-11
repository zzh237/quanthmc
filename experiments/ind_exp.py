
import os
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# from web.app import * 
import shutil
from algo.hamiltorch.util import set_random_seed
from experiments.ui_args import * 
set_random_seed()
args = create_args() #as long as the compiler see the parse arg, it starts to receive args from the terminal


from data import *     
from experiments import *

from model.vqc_layers import * 
from algo.hmc import * 
from algo.sghmc import *
from algo.sgd import * 
from algo.adam import * 
from quant_architecture.variational_qunt_circuit import * 


colordict = {'hmc':'green','nuts':'blue'}

def main()->dict:
    if args.data_name == 'cancer':
        run_exp(cancer_exp(), args)
    if args.data_name == 'iris':
        run_exp(iris_exp(),  args)
    if args.data_name == 'mnist':
        run_exp(mnist_exp(), args)
    if args.data_name == 'mnist_two_target':
        run_exp(mnist_two_target_exp(), args)
    if args.data_name == 'digits':
        run_exp(digits_exp(), args)
    if args.data_name == 'wine':
        run_exp(wine_exp(), args)
    
def run_exp(exp_interface, args)->dict:
    
    if args.data_name == 'cancer':
        exp_data = cancer_data(args)
    if args.data_name == 'iris':
        exp_data = iris_data(args)
    if args.data_name == 'mnist':
        exp_data = mnist_data(args)
    if args.data_name == 'mnist_two_target':
        exp_data = mnist_data(args)
    if args.data_name == 'digits':
        exp_data = digits_data(args)
    if args.data_name == 'wine':
        exp_data = wine_data(args)
    
    data = data_generator(exp_data).data
    data_name = repr(exp_data) 
    args.data_name = data_name 
    exp_interface.feed_data(data) #here we give the model data 

    
    exp_interface.prepare_exp(args)
    model = exp_interface.prepare_model() #here we initilize the model 

    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    args.n_params = count_parameters(model)

    if args.algo_name == 'SGHMC':
        algo = sghmc(args, model, data)
    if args.algo_name == 'HMC':
        algo = hmc(args, model, data)
    if args.algo_name == 'SGD':
        algo = sgd(args, model, data)
    if args.algo_name == 'Adam':
        algo = adam(args, model, data)
    
    
    algo_name = repr(algo)

    start_time = datetime.now()
    args.out_dir = os.path.join('result', data_name, algo_name, args.model_name, start_time.strftime("%Y%m%d-%H%M%S"))
    
    
    err, best_err, loss = exp_interface.apply_algorithm(algo)
    
    
    time_duration = datetime.now() - start_time
    
    result = {
        'err':err,
        'best_err':best_err,
        'loss':loss,
        'time': time_duration,
        'train_data':args.N_tr,
        'num_params':args.n_params,
        'args':args,
        'data_name': data_name,
        'algo_name': args.algo_name,
        'model_name': args.model_name
        }
    if args.save_result:     
        output_path = os.path.join(args.out_dir, 'sample_result.npy')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, result)
    exp_name = '_'.join([result['data_name'],result['algo_name'],result['model_name']])
    if args.draw_result:
        path1 = draw('test_err', result['err'], result['args'].out_dir, result['time'], result['train_data'], result['num_params'], exp_name, best_err=result['best_err'])
        # shutil.copyfile(path1, os.path.join('web/static/{}_err.jpg'.format(exp_name)))
        path1 = draw('Negative_Log-Likelihood', result['loss'], result['args'].out_dir, result['time'], result['train_data'], result['num_params'], exp_name)
        # shutil.copyfile(path1, os.path.join('web/static/{}_loss.jpg'.format(exp_name)))
        # copy_and_overwrite(args.out_dir, 'web/static/') #copy the whole directory to the location
        #run_app().run(host='127.0.0.1',port=5000, debug=False)
    return result

def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)


def draw(metric_name, metrics, dir, exp_time, exp_data_size, exp_param_size, exp_name, **kargs)->str:
    fig, ax = plt.subplots(1, 1, sharey=True, figsize = (10,7))
    if 'best_err' in kargs:
        best_err = kargs['best_err']
    size = len(metrics)
    ax.plot(metrics, marker='o', label=exp_name,color='blue') 
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.05,
                     box.width, box.height * 0.95])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True, ncol=3)
    ax.set_xlabel('Iteration number',fontsize=10)
    ax.set_ylabel(metric_name,fontsize=10)
    title = "{}".format(exp_name)
    ax.set_title(title, fontsize=15, fontweight='bold')
    fig.text(0.7, 0.3, "algorithm time:{:.3f}".format(exp_time.total_seconds()), va='center', rotation='horizontal')
    fig.text(0.7, 0.35, "train data size:{:.3f}".format(exp_data_size), va='center', rotation='horizontal')
    fig.text(0.7, 0.4, "parameters size:{:.3f}".format(exp_param_size), va='center', rotation='horizontal')
    if metric_name == 'test_err':
        if 'HMC' in exp_name:
            fig.text(0.65, 0.25, "{}:{:.3f}".format(metric_name, best_err),va='center',rotation='horizontal')
        else:
            fig.text(0.65, 0.25, "{}:{:.3f}".format(metric_name, metrics[-1]),va='center',rotation='horizontal')
    
    else:
        fig.text(0.65, 0.25, "{}:{:.3f}".format(metric_name, metrics[-1]),va='center',rotation='horizontal')
    
    
    path = os.path.join(dir, '{}_{}.png'.format(exp_name,metric_name))
    plt.savefig(path)
    return path

if __name__ == "__main__":
    main()

