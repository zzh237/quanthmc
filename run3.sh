
##this is the experiment 
cd /zzproject/quanthmc
export PYTHONPATH=$PYTHONPATH:/zzproject/quanthmc
for n in 0.5
do
for p in 1 2 4 6 8
do
    python3 experiments/ind_exp.py --algo_name HMC --data_name wine --model_name Quant --tr_ratio $n --q_depth $p  --step_size 0.0003 --L 10 --num_samples 100 
done
done
