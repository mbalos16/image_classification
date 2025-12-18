# Import libraries.
import argparse
import os
import random as rd
import time
from functools import partial

import kagglehub
import pytorch_warmup as warmup
import torch
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision.datasets import ImageFolder

from models.alexnet import AlexNet
from models.baseline_cnn import BaselineCNN
from models.densenet import DenseNet
from models.inceptionet import GoogLeNet
from models.lenet import LeNet, LeNetMod, LeNetModNorm, LeNetModNorm2
from models.resnet import ResNet
from models.vit import Vit
from trainer import train

MODEL_LIBRARY = {
    "baseline_cnn": BaselineCNN,
    "lenet": LeNet,
    "lenet_mod": LeNetMod,
    "lenet_mod_norm": LeNetModNorm,
    "lenet_mod_norm2": LeNetModNorm2,
    "alexnet": AlexNet,
    "inceptionet": GoogLeNet,
    "resnet": ResNet,
    "densenet": DenseNet,
    "vit": Vit,
}


def parse_args():
    parser = argparse.ArgumentParser(description="An example script")

    parser.add_argument(
        "-m",
        "--model_name",
        required=True,
        type=str,
        help="The name of the model you want to train.",
    )

    parser.add_argument(
        "-e",
        "--epocs",
        type=int,
        required=False,
        default=2000,
        help="Number of epocs to train the model.",
    )

    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        required=False,
        default=0.00001,
        help="The learning rate used for the optimizer.",
    )

    parser.add_argument(
        "-wp",
        "--warmup_period",
        type=int,
        required=False,
        default=2000,
        help="The warmup period needed to warm up the learning rate. Makes hyperparameter tuning more robust while improving the final performance.",
    )

    args = parser.parse_args()
    return args


def one_hot_encoding_labels(label, num_classes):
    """Transform the labels in one_hot_encoding."""
    one_hot_encoding_labels = torch.nn.functional.one_hot(
        torch.tensor(label),
        num_classes=num_classes,
    )
    return one_hot_encoding_labels


def get_dataloaders():
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

    # Data transformation and one hot encoding
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(p=0.5)
        ]
    )

    train_dataset = ImageFolder(
        root=train_path_img,  # Path to the dataset root directory
        transform=transform,  # Apply the defined transformations
        target_transform=partial(one_hot_encoding_labels, num_classes=6),
    )

    validation_dataset = ImageFolder(
        root=validation_path_img,  # Path to the dataset root directory
        transform=transform,  # Apply the defined transformations
        target_transform=partial(one_hot_encoding_labels, num_classes=6),
    )

    # A dictionary is defined to eazily switch between training and validation datasets during training.
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=8, shuffle=True),
        "val": DataLoader(validation_dataset, batch_size=8, shuffle=False),
    }

    return dataloaders


def main():
    args = parse_args()

    # Instantiate the model
    if not args.model_name in MODEL_LIBRARY:
        raise ValueError(
            f"The model {args.model_name} is not available. Please choose one of {MODEL_LIBRARY.keys()}."
        )
    model = MODEL_LIBRARY[args.model_name]()

    dataloaders = get_dataloaders()
    epochs = args.epocs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    warmup_scheduler = warmup.LinearWarmup(optimizer, args.warmup_period)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # Define the name to be used for the logdir folder for each model
    year = time.gmtime().tm_year
    month = time.gmtime().tm_mon
    day = time.gmtime().tm_mday
    model_log_name = f"{year}_{month}_{day}_{args.model_name}"
    writer = SummaryWriter(
        os.path.join("runs", model_log_name)
    )  # Tensor Board Logs Class

    train(
        epochs=epochs,
        dataloaders=dataloaders,
        optimizer=optimizer,
        device=device,
        model=model,
        writer=writer,
        warmup_scheduler=warmup_scheduler,
    )


if __name__ == "__main__":
    main()
