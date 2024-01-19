import torch.cuda
import torch.nn as nn
import torch.optim as optim

from dataset import LFWDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval_metrics import calculate_segmentation_metrics
from log import log_predictions
from model.UNet import UNet
import wandb
from model.model_checkpoint import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(num_checkpoints=5, decreasing_metric=False, checkpoint_dir="/Users/andrei")


def train(train_loader, val_loader, model, optimizer, criterion, current_epoch, total_epochs, device, validation_table, num_classes, wandb_log=True):
    model.train()  # Set the model to training mode

    # Customized tqdm progress bar with correct epoch display
    pbar = tqdm(train_loader, desc=f"Epoch {current_epoch + 1}/{total_epochs}", unit='images',
                unit_scale=train_loader.batch_size, total=len(train_loader))

    running_loss = 0.0

    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()  # Zero the optimizer's gradients
        outputs = model(images)  # Perform the forward pass
        loss = criterion(outputs, masks)  # Calculate the loss
        running_loss += loss.item()

        loss.backward()  # Calculate gradients (backpropagation)
        optimizer.step()  # Adjust the model's weights

        # Update the progress bar with the running loss
        pbar.set_postfix({"Training Loss": f"{running_loss / (batch_idx + 1):.4f}"})

    if wandb_log:
        wandb.log({"training_loss": running_loss / len(train_loader)})

    for images, masks in val_loader:
        log_predictions(validation_table, model, images, masks, device)
        break  # Log one batch per epoch for demonstration

    # Evaluate on validation set after each epoch
    validate(val_loader, model, criterion, device, current_epoch, num_classes)


def validate(val_loader, model, criterion, device, current_epoch, num_classes):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    total_mpa = 0.0
    total_miou = 0.0
    total_fwiou = 0.0

    with torch.no_grad():  # Disable gradient calculation
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            mpa, miou, fwiou = calculate_segmentation_metrics(outputs, masks, num_classes)
            total_mpa += mpa
            total_miou += miou
            total_fwiou += fwiou

        avg_val_loss = val_loss / len(val_loader)
        avg_mpa = total_mpa / len(val_loader)  # Average MPA over validation set
        avg_miou = total_miou / len(val_loader)
        avg_fwiou = total_fwiou / len(val_loader)
        wandb.log({"validation-loss": avg_val_loss, "mean-pixel-accuracy": avg_mpa, "mean-iou": avg_miou,
                   "mean-fwiou": avg_fwiou})
        checkpoint_callback(model, current_epoch, avg_fwiou)

        print(
            f"Validation Loss: {avg_val_loss:.4f}\nMean Pixel Accuracy: {avg_mpa:.4f}\nMean IoU Accuracy: {avg_miou:.4f}\nMean FWIoU Accuracy: {avg_fwiou:.4f}\n")


if __name__ == '__main__':
    wandb.init(project="semantic-segmentation-model")
    learning_rate = 0.0001
    epochs = 10
    batch_size = 8
    wandb.config.update({"learning_rate": learning_rate,
                         "epochs": epochs,
                         "batch_size": batch_size})
    validation_table = wandb.Table(columns=["Image", "Prediction", "Ground Truth"])

    device = "mps"  # Device for computation

    # Load and prepare the training data
    train_dataset = LFWDataset(download=False, base_folder='lfw_dataset', split_name="train", transforms=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, sampler=None,
                              num_workers=0)

    # Load and prepare the validation data
    val_dataset = LFWDataset(download=False, base_folder='lfw_dataset',
                             split_name="val", transforms=None)  # Ensure you have a validation split
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, sampler=None,
                            num_workers=0)

    encoder_channels = [3, 64, 128, 256, 512]
    decoder_depths = [256, 128, 64]
    num_classes = 3  # Number of classes in the segmentation problem

    # Initialize the U-Net model
    model = UNet(encoder_channels, decoder_depths, num_classes)
    model = model.to(device)  # Move the model to the specified device

    # Set up the optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Define the loss function - CrossEntropyLoss for multi-class segmentation
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        train(train_loader, val_loader, model, optimizer, criterion, epoch, epochs, device, validation_table)

    # Log metrics and table to wandb
    wandb.log({"validation_predictions": validation_table})
    wandb.finish()
