cd /zzproject/quanthmc
export PYTHONPATH=$PYTHONPATH:/zzproject/quanthmc
for n in 0.5
do
for p in 1 2 4 6 8 
do
	python3 experiments/ind_exp.py --algo_name Adam --data_name wine --model_name Quant  --tr_ratio $n --q_depth $p 
done
done
