for n in 0.1
do
for m in 0 
do
	python experiments/ind_exp.py --algo_name HMC --data_name iris --model_name Quant --mlp_depth $m  --tr_ratio $n --step_size 0.1 --L 20 --num_samples 100 
done
done
