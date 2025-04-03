#!/bin/bash
apt-get update
apt-get upgrade -y
apt-get install -y curl
apt-get install -y git
apt-get install -y zip

curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -bfp /usr/local
rm -f Miniconda3-latest-Linux-x86_64.sh

PATH="/usr/local/miniconda3/bin":$PATH
conda init bash
echo 'conda activate base' >> ~/.bashrc

conda install -y python=3.12 make conda-forge::cxx-compiler conda-forge::numpy=1.26.4
conda install -y conda-forge::scikit-learn
conda install -y conda-forge::seaborn conda-forge::pandas conda-forge::polars conda-forge::pyarrow
conda install -y conda-forge::jupyterlab conda-forge::tqdm conda-forge::ipywidgets
conda install -y conda-forge::lightgbm

pip3 install torch==2.4.0 torchvision torchaudio
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip3 install "ray[tune]" kmeans-pytorch transformers
pip3 install plotly

pip install -e ./RecBole --verbose
