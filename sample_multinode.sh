# Runs distributed training on the cluster (multiple nodes).

# define the config path
cfg=configs/main/avid/kinetics/cross_16x112x112.yaml

# define the nodes
MASTER_NODE_IP=10.141.0.2

# run on node 0
ssh -n -f node402 "echo 'I am in Node 402' > sample.txt"
echo "I am in Node 402"
export PYTHONPATH=/var/scratch/pbagad/projects/AVID-CMA/
# python /var/scratch/pbagad/projects/AVID-CMA/main-avid.py $cfg \
#     --dist-url tcp://$MASTER_NODE_IP:1234 \
#     --multiprocessing-distributed \
#     --world-size 2 \
#     --rank 0
exit

# run on node 1
ssh node403
echo "I am in Node 403"
export PYTHONPATH=/var/scratch/pbagad/projects/AVID-CMA/
# python /var/scratch/pbagad/projects/AVID-CMA/main-avid.py $cfg \
#     --dist-url tcp://$MASTER_NODE_IP:1234 \
#     --multiprocessing-distributed \
#     --world-size 2 \
#     --rank 1
exit
