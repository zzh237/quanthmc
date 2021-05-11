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
for n in 0.01 0.03 0.05 0.07
do
        echo "run second"
        python3 experiments/ind_exp.py --algo_name HMC --data_name mnist_two_target --model_name mlp  --tr_ratio $n --step_size 0.001 --L 20 --num_samples 100
done

done 