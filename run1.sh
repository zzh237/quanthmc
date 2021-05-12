# this is the experiment for adam optimizer

cd /zzproject/quanthmc

export PYTHONPATH=$PYTHONPATH:/zzproject/quanthmc

for i in 1 
do 

# for n in 0.01 0.03 0.05 0.07
# do
# for m in 0 1
# do
#         echo "run first"
#         python3 experiments/ind_exp.py --algo_name Adam --data_name mnist_two_target --model_name mlp --mlp_depth $m  --tr_ratio $n 
# done
# done

# run the adam mlp hmc experiment
for n in 0.7 
do
for m in 0 
do
for s in 0.01
do
for l in 20
do 
        echo "run second"
        python3 experiments/ind_exp.py --algo_name HMC --data_name cancer --model_name mlp  --tr_ratio $n --mlp_depth $m --step_size $s --L $l --num_samples 100
done
done  
done
done 
done 