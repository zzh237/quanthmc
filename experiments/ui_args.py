import argparse 
import numpy as np 
import torch 

def create_args()->dict:
    """[summary]
    Returns:
        dict: [a dictionary contains the args and non args parameters]
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--algo_name', default='HMC', type=str, 
                                help='？')
    
    parser.add_argument('--model_name', default='Quant', type=str, 
                                help='？')

    parser.add_argument('--data_name', default='iris', type=str, 
                                help='？')
    
    # Define the neural network depth and complexity

    parser.add_argument('--mlp_depth', default=2, type=int, help="?")
    parser.add_argument('--mlp_width', default=128, type=int, help="?")

    ## these parameters are created under the running process 
    parser.add_argument('--n_params', default=10, type=int, help="?")
    parser.add_argument('--tr_ratio', default=0.1, type=float, help="?")
    parser.add_argument('--feature_dim', default=4, type=int, help="?")
    parser.add_argument('--target_class', default=1, type=int, help="?")


    
    # DEFINE THE QUANTUM CIRCUIT

    parser.add_argument('--n_qubits', default=4, type=int, 
                        help='number of qubits')

    parser.add_argument('--q_depth', default=2, type=int,
                        help='Depth of the quantum circuit (number of variational layers)')
    
    ## nuts parameter
    parser.add_argument('--q_delta', default=0.01, type=float,
                        help='Initial spread of random quantum weights')
    

    parser.add_argument('--device', default='cpu', type=str, 
                        help='device avaialbe')
    
    
    ## hmc parameter
    parser.add_argument('--step_size', default=0.1, type=float, 
                        help='leapfrog step_size used')
    
    parser.add_argument('--L', default=20, type=int, 
                        help='L leapfrog steps envolved')
    
    parser.add_argument('--num_samples', default=100, type=int, 
                        help='hmc samples returned')
    
    parser.add_argument('--tau_out', default=1., type=float, 
                        help='?')

    parser.add_argument('--tau', default=1., type=float, 
                        help='?')

    ## data parameter N_tr = 10#50
    parser.add_argument('--N_tr', default=120, type=int, 
                        help='?')
    
    parser.add_argument('--N_val', default=30, type=int, 
                        help='?')

    parser.add_argument('--batch_size', default=40, type=int, help="?")

    ### sghmc parameters
    parser.add_argument('--epochs', type=int, nargs='?', action='store', default=100,
                    help='How many epochs to train. Default: .')
    parser.add_argument('--sample_freq', type=int, nargs='?', action='store', default=2,
                    help='How many epochs pass between saving samples. Default: 2.')
    parser.add_argument('--burn_in', type=int, nargs='?', action='store', default=20,
                    help='How many epochs to burn in for?. Default: 20.')
    parser.add_argument('--lr', type=float, nargs='?', action='store', default=1e-3,
                    help='learning rate. I recommend 1e-2. Default: 1e-2.')

    parser.add_argument('--grad_std_mul', default=20, type=int, help="?")
    
    
    parser.add_argument('--flat_ims', default=False, type=bool, 
                        help='flat the data or not')

    parser.add_argument('--log_interval', default=1, type=int, 
                        help='log interval')                    
    # log_interval = 1
    # nb_its_dev = log_interval
    parser.add_argument('--resample_its', default=50, type=int, 
                            help='？')

    parser.add_argument('--N_saves', default=100, type=int, 
                                help='？')

    parser.add_argument('--resample_prior_its', default=15, type=int, 
                                help='？')

    
    
    parser.add_argument('--re_burn', default=1e8, type=int, 
                                help='？')

    # re_burn = 1e8
    
    
    
    
    parser.add_argument('-od', '--out_dir', default='', type=str, 
                        help='output directory')
    parser.add_argument('-s','--save_result', default=True, type=bool, 
                        help='save result to local or not')
   
    parser.add_argument('-draw','--draw_result', default=True, type=bool, 
                        help='draw result or not')
    
    
    
    # distributed processing
    parser.add_argument('--num_workers', default=3, type=int, 
                            help='？')

    args = parser.parse_args()
    
    # args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = 'cpu'
    if args.device == 'cuda':
        print("########: GPU IS Open!")

    if args.device == 'cuda':
        args.num_workers = 0
    else:
        args.num_workers = 3

    


    # theta0 = np.random.normal(0, 1, args.D)
    # mean = np.zeros(args.D)
    # cov = np.asarray([[1, 1.98], 
    #                 [1.98, 4]])
    # params = {'distribution':{'theta0':theta0, 'mean':mean,'cov':cov}, 'args':args}
    return args






