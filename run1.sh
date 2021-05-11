# this is the experiment for adam optimizer

cd /zzproject/quanthmc

export PYTHONPATH=$PYTHONPATH:/zzproject/quanthmc

for n in 0.1 0.3 0.5 0.7
do
for m in 0 1
do
        python3 experiments/ind_exp.py --algo_name Adam --data_name wine --model_name mlp --mlp_depth $m  --tr_ratio $n 
done
done

## run the adam mlp hmc experiment
# for n in 0.1 0.3 0.5 0.7
# do
#         python3 experiments/ind_exp.py --algo_name HMC --data_name wine --model_name mlp  --tr_ratio $n --step_size 0.0003 --L 10 --num_samples 100
# done

