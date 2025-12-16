# Import libraries.
import os
import random as rd
from functools import partial

import kagglehub
import pytorch_warmup as warmup
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision.datasets import ImageFolder

from models.alexnet import AlexNet
from models.custom_cnn import CustomModel
from models.lenet import LeNet, LeNetMod, LeNetModNorm, LeNetModNorm2
from models.resnet import ResNet
from trainer import train

# Load the data.
path = kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database")
validation_path_img = os.path.join(path, "NEU-DET", "validation", "images")
validation_path_ann = os.path.join(path, "NEU-DET", "validation", "annotations")
train_path_img = os.path.join(path, "NEU-DET", "train", "images")
train_path_ann = os.path.join(path, "NEU-DET", "train", "annotations")

# Check the diferent categories available in the training dataset and print them in the console.
training_categories = os.listdir(train_path_img)
print(
    f"The training dataset has {len(training_categories)} categories which are: {training_categories}"
)

# Print 3 images for each category in the dataset
for category in training_categories:
    # Plot 3 random images from the training set for each category
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1 row, 3 columns
    for idx in range(3):
        random_image = rd.randint(1, 240)

        # Plot the image
        image_path = os.path.join(
            train_path_img, category, f"{category}_{random_image}.jpg"
        )
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        axes[idx].imshow(image)
        axes[idx].set_title(f"{category}_{random_image}", loc="left")

    plt.suptitle(f"Category: {category}\n", fontsize=14)
    plt.tight_layout()
    plt.show()

    # Check if all images in the category have the same dimensions
    cat_len = len(os.listdir(os.path.join(train_path_img, category)))
    initia_width, initial_height = 0, 0
    for i in range(cat_len):
        image_path = os.path.join(train_path_img, category, f"{category}_{i+1}.jpg")
        with Image.open(image_path) as im:
            width, height = im.size
            if i == 0:
                initia_width = width
                initial_height = height
            else:
                if width != initia_width or height != initial_height:
                    print(
                        f"Image {i+1} in category {category} has different dimensions: {width}x{height} vs {initia_width}x{initial_height}"
                    )
    print(
        f"Image dimmensions: {initia_width} x {initial_height} in category {category}"
    )
    print(f"The category {category} in training has {cat_len} images.\n\n")

# Check the different categories in the validation dataset
validation_categories = os.listdir(validation_path_img)
print(
    f"The validation dataset has {len(validation_categories)} categories which are: {validation_categories}"
)
print(
    f"The validation dataset has: {len(validation_categories) * len(os.listdir(os.path.join(validation_path_img, category)))} images"
)

# Data transformation and one hot encoding
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(p=0.5)
    ]
)
