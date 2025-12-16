import torch


def get_lr(optimizer):
    """A function that extracts the learning rate from the optimizer.

    Args:
        optimizer (_type_): OPtimizer used

    Returns:
        _type_: The learning rate used.
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train(
    model, device, optimizer, dataloaders, writer, warmup_scheduler=None, epochs=2500
):

    # Define a training step to keep track of the logs for each step.
    training_step = 0

    # Define the loss.
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoc_metrics = {
            "train": {"loss": 0, "accuracy": 0, "count": 0},
            "val": {"loss": 0, "accuracy": 0, "count": 0},
        }
        # print(f"======================= Epoch: {epoch} =======================")

        for phase in ["val", "train"]:
            # print(f"======================= Phase: {phase} ======================= ")
            for images, labels_one_hot in dataloaders[phase]:
                batch_size = len(labels_one_hot)
                optimizer.zero_grad()  # Reset the gradients to zero before each batch
                with torch.set_grad_enabled(phase == "train"):
                    output = model(images.to(device))
                    loss = loss_fn(output, labels_one_hot.float().to(device))
                    predicted_labels = torch.argmax(output, dim=1)
                    labels = torch.argmax(labels_one_hot, dim=1)
                    correct_preds = labels.to(device) == predicted_labels
                    accuracy = (correct_preds).sum() / batch_size

                if phase == "train":
                    model.train()
                    training_step += 1
                    loss.backward()
                    optimizer.step()

                    # Add the loss and epoch to the tensorboard
                    writer.add_scalar("Train/Loss", loss, training_step)
                    writer.add_scalar("Train/Accuracy", accuracy, training_step)
                    writer.add_scalar("Train/LR", get_lr(optimizer), training_step)

                    if warmup_scheduler is not None:
                        with warmup_scheduler.dampening():
                            pass
                else:
                    model.eval()
                epoc_metrics[phase]["loss"] += loss.item()
                epoc_metrics[phase]["accuracy"] += accuracy.item()
                epoc_metrics[phase]["count"] += 1

            ep_loss = epoc_metrics[phase]["loss"] / epoc_metrics[phase]["count"]
            ep_accuracy = epoc_metrics[phase]["accuracy"] / epoc_metrics[phase]["count"]

            writer.add_scalar(f"{phase.capitalize()}/EpochLoss", ep_loss, training_step)
            writer.add_scalar(
                f"{phase.capitalize()}/EpochAccuracy", ep_accuracy, training_step
            )

            print(f"Loss: {ep_loss}, Accuracy: {ep_accuracy}\n")
            optimizer.step()
    writer.flush()  # Tensorboard: Add all the pending events
