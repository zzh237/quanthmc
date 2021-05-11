

import matplotlib.pyplot as plt
import numpy as np 
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
import os 
from mpl_toolkits.mplot3d import Axes3D

def draw_ham(path):
    result = np.load(path,allow_pickle=True).item(0)
    print('zz')
    # result = {
    #     'err':err,
    #     'best_err':best_err,
    #     'loss':loss,
    #     'time': time_duration,
    #     'train_data':args.N_tr,
    #     'num_params':args.n_params,
    #     'args':args,
    #     'data_name': data_name,
    #     'algo_name': args.algo_name,
    #     'model_name': args.model_name
    #     }
    draw_scatter(result)


def draw_2d(result):
    fig = plt.figure()
    ax = Axes3D(fig)
    h, p, k = result['hamitonian'], result['potential'],result['kinetic'] 
    h = np.array(h)
    p = np.array(p)
    k = np.array(k)
    size = len(h)
    ax.plot_surface(h,p,k, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    

    
    dir_list = os.path.split(os.path.dirname(path))
    dir = dir_list[:-1][0]
    img_path = os.path.join(dir, "2d_energy.png")
    plt.savefig(img_path)

def draw_scatter(result)->str:
    
    # result['new_hamitonian'].append(ham)
    # result['new_potential'].append(new_potential)
    # result['new_kinetic'].append(new_kinetic)
    
    fig, ax = plt.subplots(1, 1, sharey=True, figsize = (10,7))
    h, p, k = result['hamitonian'], result['potential'],result['kinetic'] 
    def f(x):
        return x.detach().numpy()
    
    h = list(map(f, h))
    p = list(map(f,p))
    k = list(map(f,k))
    size = len(h)
    
    
    ax.scatter(p, k, marker='.',color='blue') 
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.05,
                     box.width, box.height * 0.95])
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True, ncol=3)
    ax.set_xlabel(r"-log(\theta|D)",fontsize=10)
    ax.set_ylabel("Kinetic", fontsize=10)
    # title = "{}".format(exp_name)
    # ax.set_title(title, fontsize=15, fontweight='bold')
    
    # fig.text(0.7, 0.3, "algorithm time:{:.3f}".format(exp_time.total_seconds()), va='center', rotation='horizontal')
    
    dir_list = os.path.split(os.path.dirname(path))
    dir = dir_list[:-1][0]
    img_path = os.path.join(dir, "2d_energy_scatter.png")
    plt.savefig(img_path)


def draw_lines(result)->str:
    
    # result['new_hamitonian'].append(ham)
    # result['new_potential'].append(new_potential)
    # result['new_kinetic'].append(new_kinetic)
    
    fig, ax = plt.subplots(1, 1, sharey=True, figsize = (10,7))
    h, p, k = result['hamitonian'], result['potential'],result['kinetic'] 
    size = len(h)
    
    ax.plot(h,  label="Hamiltonian",color='blue') 
    ax.plot(p,  label=r"-log(\theta|D)",color='orange')
    ax.plot(k, label=r"Kinetic",color='red')
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.05,
                     box.width, box.height * 0.95])
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True, ncol=3)
    ax.set_xlabel('Samling Size',fontsize=10)
    
    # ax.set_ylabel(,fontsize=10)
    # title = "{}".format(exp_name)
    # ax.set_title(title, fontsize=15, fontweight='bold')
    
    # fig.text(0.7, 0.3, "algorithm time:{:.3f}".format(exp_time.total_seconds()), va='center', rotation='horizontal')
    
    dir_list = os.path.split(os.path.dirname(path))
    dir = dir_list[:-1][0]
    img_path = os.path.join(dir, "2d_energy.png")
    plt.savefig(img_path)
    


if __name__ == "__main__":
    path = "/Users/zz/Downloads/zzprojects/quanthmc/result/iris/hmc/Quant/20210509-211637/hmc_result.npy"
    # path = "/Users/zz/Downloads/zzprojects/quanthmc/result/iris/hmc/mlp/20210506-102505/hmc_result.npy"
    draw_ham(path)
