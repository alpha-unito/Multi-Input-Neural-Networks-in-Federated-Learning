# Multi-Input-Neural-Networks-in-Federated-Learning

This repository contains the code used to train a Multi-Input Neural Network (NN) in a centralized (standard Deep Learning) and in a federated setting using [PyTorch](https://pytorch.org/) and the Intel [OpenFL](https://openfl.readthedocs.io/en/latest/index.html) framework. The Multi-Input NN consists of a [ResNet-18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) concatenated with a Multilayer Perceptron (MLP).

Extensive experiments have been executed in a centralized and in a federated scenarios, testing a Multi-Input NN with a ResNet-18 or with an MLP, over two datasets:
- [CoViD-19 CXR](https://ai4covid-hackathon.it/), trained on the [HPC4AI](https://hpc4ai.unito.it/documentation/) cluster at the University of Turin (node: 8 cores per CPU, AMD EPYC-IPBP, 1 NVIDIA A40 GPU));
- [ADNI](https://adni.loni.usc.edu/), trained on...

## Usage

To run the experiment as is, clone [this]([https://github.com/alpha-unito/streamflow-fl](https://github.com/CasellaJr/Multi-Input-Neural-Networks-in-Federated-Learning)) repository and use the following:
- For CoViD-19 CXR dataset in centralized version:
- ```
  1. Put in the same folder: 'trainANDtest.xls', the folder containing the images 'DATASET', and the notebooks.
  2. Run all the cells of the notebooks (Multi-input, only images and only text).
  ```
- For CoViD-19 CXR dataset in federated version:
- ```
  1. Install [OpenFL](https://openfl.readthedocs.io/en/latest/index.html)
  2. Put in the same folder: the folder containing the images 'DATASET', director, envoy and the workspace.
  3. Open a terminal for the director and one for each envoy.
  4. `./start_director.sh`
  5. `./start_envoy.sh`
  6. Run all the cells of the notebook in the workspace.
  7. For reproducibility, do 5 runs changing the variable `myseed` from 0 to 4 and then calculate mean and standard deviation of the best aggregated model.
  ```
- For ADNI dataset in centralized version:
- For ADNI dataset in federated version:



## Contributors

Bruno Casella <bruno.casella@unito.it>  
Walter Riviera <walter.riviera@intel.com>  
Marco Aldinucci <marco.aldinucci@unito.it>  
Gloria Menegaz <gloria.menegaz@gmail.com>


