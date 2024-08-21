# Machine Vision Toolkit

MVT (Machine Vision Toolkit) is a collection of utilities and scripts to facilitate the training of machine learning models and write computer vision algorithms. Among others, the code allows to create image classifiers, feature extractors, data generators, train deep neural networks and plot graphs.

## Features

### Tasks
* [x] Classification
* [ ] Detection
* [ ] Segmentation
* [ ] Landmark estimation

### Models
* [x] Everything avaliable in `timm` package
* [x] YOLO-like, ie. backbone + PANet + head

### Datasets
* [x] CIFAR 10
* [x] CIFAR 20
* [x] CIFAR 100
* [x] COCO
* [ ] COCO Whole Body
* [x] Imagenet
* [x] Oxford IIIT Pet
* [ ] Cityscapes
* [ ] MNIST
* [ ] CINIC-10

### Training
* [ ] Learning rate schedulers
    * [x] Constant (no-decay)
    * [x] Step decay
    * [x] Linear decay
    * [x] Cosine decay
    * [ ] Exponential decay
    * [ ] Cyclical decay
    * [ ] Warmup
* [x] Parallelization
    * [x] Multi-GPU
    * [x] Multi-node
* [x] Data augmentation

## Installation

Create a new conda environment and install the required packages.

```bash
conda create -n mvt
conda activate mvt
conda install python=3.7

pip install .
```

If you plan to modify the code, use `-e` flag to install the package in editable mode.  
Add `--force` flag to force the setup installation if required modules have changed.

```bash
pip install -e . --force
```

## Getting started (by examples)

At the time of writing, training can be done by running the `bin/train` script.  As an example, the following command is to train an image classifier: the network is a ResNet18, while the dataset is CIFAR-10. Outputs will be saved in the `results/rn18_c10` folder.

```bash
./bin/train --model_name resnet18 --dataset_root ~/data/datasets/ --dataset_name cifar10 --output results/rn18_c10
```

All settings have a default value but `--output`, which is required to be set by the user.  
In this example, the model and dataset names have been set for clarity.  

#### Dataset root
Each dataset is expected to be saved in specific folder, contained in the path set with the `--dataset_root` argument.  
In the above example, the root folder is `~/data/datasets/`, where CIFAR10 is expected to be found as `~/data/datasets/cifar/cifar-10-batches-py`.
This is specific for CIFAR10 (for example, CIFAR100 is expected ti be in `~/data/datasets/cifar/cifar-100-python`).

Check the dataset class in `mvt/datasets.py` if you struggle with dataset paths (by design, each dataset expects a specific path).

## Distributed training

It is possibile to reduce training time thanks to distributed training, specifically `torchrun` is used to allocate the right resources.

```bash
# train on 2 gpus, both on a single machine
torchrun --nnodes=1 --nproc_per_node=2 ./bin/train --output results

# train on 8 gpus, 2 machines with 4 gpus each
torchrun --nnodes=2 --nproc_per_node=4 ./bin/train --output results
```

## Notes

Writing styles and conventions are based on [this Python guide](https://docs.python-guide.org/writing/structure/).

## License

[Machine Vision Toolkit](https://github.com/mcarletti/machinevisiontoolkit) © 2024 by [Marco Carletti](https://www.marcocarletti.it/) is licensed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/?ref=chooser-v1).

CC BY-SA 4.0  
**Attribution-ShareAlike 4.0 International**  
This license requires that reusers give credit to the creator. It allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, even for commercial purposes. If others remix, adapt, or build upon the material, they must license the modified material under identical terms.

BY: Credit must be given to you, the creator.  
SA: Adaptations must be shared under the same terms.
