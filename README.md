# CON(e)VOLUTION - From LeNet to Vision Transformers
Convolutional Neural Networks Architectures for Image Classification
`Maria Magdalena Balos`

This repository accompanies the Medium article **“CON(e)VOLUTION – The evolution of deep learning vision architectures”**.

The article shows my journey in implementating and experimenting with classical and modern vision architectures — from early CNNs like LeNet and AlexNet to Inception, ResNet, DenseNet, and Vision Transformers (ViT) — applied to an image classification task.

The purpose of this project is educational and exploratory:
understand how vision architectures evolved, what problems they were designed to solve, and how their design choices affect learning behavior — not to propose new models or chase state-of-the-art results.


The architectures explored within this project are:
* BaselineCNN
* LeNet
* LeNetMod
* LeNetModNorm
* LeNetModNorm2
* AlexNet
* GoogLeNet
* ResNet
* DenseNet
* Vit

Read the full article on Medium: [CON(e)VOLUTION](https://medium.com/@mariabalos16/con-e-volution-a-walkthrough-from-lenet-to-vision-transformers-4f319bb0b2b7).


## Notes you might want to know about this project

* Models are implemented in PyTorch.
* Experiments use the [Neu Surface Defects Dataset](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database).
* AI tools were used for proofreading; all code and content are original.


## Getting Started
### Setup for training a model
1. use `git clone` to clone the repository in your local directory.
2. next, use the command `cd image_classification` to go inside the project's directory.
3. use the command `uv sync --upgrade` to ensure all project dependencies are installed and up-to-date with the lockfile.
4. activate the environment with the `source .venv/bin/activate` command.
5. train a model wirh the following example:
``` python
python main.py \
    --model_name lenet \
    --epocs 1000 \
    --learning_rate 0.0001 \
    --warmup_period 2000 \
```

Thank you for taking the time to read the Medium story and this repository. As always, feedback, corrections, or suggestions are deeply appreciated.