# Runs distributed training on the cluster (multiple nodes).

# define the config path
cfg=configs/main/avid/kinetics/cross_16x112x112.yaml

# define the nodes
MASTER_NODE_IP=10.141.0.2

# run on node 0
python main-avid.py $cfg --dist-url tcp://$MASTER_NODE_IP:1234 --multiprocessing-distributed --world-size 2 --rank 0

# run on node 1
python main-avid.py $cfg --dist-url tcp://$MASTER_NODE_IP:1234 --multiprocessing-distributed --world-size 2 --rank 1
