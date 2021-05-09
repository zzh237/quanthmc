cd /zzproject/quanthmc
apt-get -y install cmake protobuf-compiler
python3 -m pip install --user --upgrade pip
# pip3 install qulacs-gpu
pip3 install pennylane-qulacs["gpu"]
pip3 install matplotlib
pip3 install -U scikit-learn
pip3 install termcolor
export PYTHONPATH=$PYTHONPATH:/zzproject/quanthmc
for n in 0.01 
do
for m in 0
do
	python3 experiments/ind_exp.py --algo_name HMC --data_name mnist_two_target --model_name Quant --mlp_depth $m  --tr_ratio $n --step_size 0.1 --L 20 --num_samples 100 
done
done
