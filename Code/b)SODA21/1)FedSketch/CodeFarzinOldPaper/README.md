# Federated Learning
This repository implements Federated Learning with Compression. This code is for NeurIPS submission and review, hence, any reuse outside this review process, or public release is prohibited.


## Getting Started
To run this code, you need to have PyTorch 1.5.0 installed with `distributed` API support. You can check it with:
```cli
import torch
torch.distributed.is_available()
```
For distributed setting, you need to have GLOO support for PyTorch. You can install the requirements for this repo using the `requirement.txt` with:
```cli
pip install -r requirement.txt
```


#### A general instruction
To run a sample code please run the following command. You may change parameters inside the run.py file.
```cli
python run.py -t lgt -n 10 -d mnist -y 0.1 -b 50 -c 10 -f local_step -s 10 -r 2 -q
```
where `-q` indicates that we want to use quantization. The `-t` option indicates the federated learning type between `scaffold`, `fedavg`, and `lgt` options. For more information you can run:
```cli
python run.py -h
```