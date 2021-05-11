
##this is the experiment 
cd /zzproject/quanthmc
export PYTHONPATH=$PYTHONPATH:/zzproject/quanthmc
for n in 0.5
do
for p in 2
do
    python3 experiments/ind_exp.py --algo_name HMC --data_name mnist_two_target --model_name Quant --tr_ratio $n --q_depth $p  --step_size 0.01 --L 20 --num_samples 100 
done
done
