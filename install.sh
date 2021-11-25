# create conda environment and install dependencies
conda create --name avid --file conda-spec-list.txt
conda activate avid

# install additional dependencies
pip install ipdb librosa pyyaml tensorboard wandb