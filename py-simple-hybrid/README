
Dependencies are

1. Cudamat

wget http://cudamat.googlecode.com/files/cudamat-01-15-2010.tar.gz
tar -xzvf cudamat-01-15-2010.tar.gz
cd cudamat
make

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'`pwd` >> ~/.bashrc.user
source ~/.bashrc.user

2. gnumpy

curl -O http://www.cs.toronto.edu/~tijmen/gnumpy.py

3. Get npmat to run on CPU

curl -O http://www.cs.toronto.edu/~ilya/npmat.py

Notes on choosing board/switching to cpu

To run on CPU

export GNUMPY_USE_GPU=no

To use a different board edit the line "gp.board_id_to_use=<gpu-id>" in sgd.py right after the gnumpy import