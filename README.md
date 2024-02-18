# ECS 189G: Deep Learning

<<<<<<< HEAD
Stage 2: First trial with PyTorch and build up a Multi-Layer Perceptron model to classify the instances in a provided dataset.  
Stage 3: Object recognition from images with convolutional neural network model.  
Stage 4: Text classification and generation with recurrent neural network models.  
Stage 5: Network embedding and node classification with graph neural network models.  


To run code for stage 3, please install torchvision.
=======
## Group Members: Aditya Seth, Brian Li, Parth Pawar, Ryan Yu

## Pre-Execution Setup
Install the following packages with pip if you do not have them already.
- pytorch
- numpy
- torchvision (needed for stage 3)
- matplotlib
-  scikit-learn & scikit-image

## Execution Instructions
1. Open up a new terminal
2. Navigate to the root directory where code, data, result, and script can be seen
![root path](./images/image-3.png)
![root contents](./images/image.png)
3. Execute the following command based on what you want to run: `python -m script.stage_(num)_script.script_(script_name)`.
   - Replace num with the stage num (i.e 3)
   - Replace script_name with the desired script (i.e cifar)

## Stage 2 Terminal Shortcuts
   - Decision Tree: `python -m script.stage_2_script.script_decision_tree`
   - MLP: `python -m script.stage_2_script.script_mlp`
   - SVM: `python -m script.stage_2_script.script_svm`

## Stage 3 Terminal Shortcuts
   - MNIST: `python -m script.stage_3_script.script_mnist`
   - CIFAR-10: `python -m script.stage_3_script.script_cifar`
     - ORL: `python -m script.stage_3_script.script_orl`
