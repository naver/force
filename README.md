# FORCE
Official implementation of FORCE algorithm from *Progressive Skeletonization: Trimming more fat from a network at initialization* (https://arxiv.org/abs/2006.09081)

## Requirements
You can create a conda environment with all the necessary libraries with the command:
    
    conda create --name myenv --file conda-env.txt

You will need to download the Tiny-Imagenet and Imagenet datasets. For Tiny-Imagenet modify the path specified in `experiments/datasets.py` For Imagenet you will need to specify it when runing the code.

## Examples

### CIFAR and Tiny Imagenet datasets
To run an experiment with CIFAR10/100 or Tiny Imagenet datasets, run:

    python train_cifar_tiny_imagenet.py --network_name vgg19 --pruning_factor 0.01 --prune_method 1 --dataset CIFAR10 --num_steps 60 --mode exp --num_batches 1
    
Alternatively, you can change the `--dataset` option with `CIFAR100` or `tiny_imagenet`. Bear in mind that for `CIFAR100` you will need to set `--num_batches` to 10 and for `tiny_imagenet` to 20. You may also change the architecture and use `resnet50` for instance. 

### Imagenet
To run an experiment with Imagenet run:

    python train_imagenet.py /path_to_dataset/ --network-name resnet50 --pruning_factor 0.05 --prune_method 1 --num_steps 60 --mode exp --num_batches 40 --epochs 90

## License

```
Copyright (c) 2020-present NAVER Corp.


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
