import torch.cuda
import torch.nn as nn
import torch.optim as optim

from dataset import LFWDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval_metrics import calculate_segmentation_metrics
from model.UNet import UNet
import wandb


def train(train_loader, val_loader, model, optimizer, criterion, current_epoch, total_epochs, device):
    model.train()  # Set the model to training mode

    # Customized tqdm progress bar with correct epoch display
    pbar = tqdm(train_loader, desc=f"Epoch {current_epoch + 1}/{total_epochs}", unit='images',
                unit_scale=train_loader.batch_size, total=len(train_loader))

    running_loss = 0.0

    for batch_idx, (images, masks) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()  # 2. Zero the optimizer's gradients
        outputs = model(images)  # 3. Perform the forward pass
        loss = criterion(outputs, masks)  # 4. Calculate the loss
        running_loss += loss.item()

        loss.backward()  # 4. Calculate gradients (backpropagation)
        optimizer.step()  # 5. Adjust the model's weights

        # Update the progress bar with the running loss
        pbar.set_postfix({"Training Loss": f"{running_loss / (batch_idx + 1):.4f}"})

    # 6. Evaluate on validation set after each epoch
    validate(val_loader, model, criterion, device)


def validate(val_loader, model, criterion, device):
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
        wandb.log({"validation-loss": avg_val_loss, "mean-pixel-accuracy": avg_mpa, "mean-iou": avg_miou, "mean-fwiou": avg_fwiou})

        print(f"Validation Loss: {avg_val_loss:.4f}\nMean Pixel Accuracy: {avg_mpa:.4f}\nMean IoU Accuracy: {avg_miou:.4f}\nMean FWIoU Accuracy: {avg_fwiou:.4f}\n")


if __name__ == '__main__':
    wandb.init(project="semantic-segmentation-model")

    device = "cuda"  # Device for computation. Using CPU because CUDA is not available

    # Load and prepare the training data
    train_dataset = LFWDataset(download=False, base_folder='lfw_dataset', split_name="train", transforms=None)
    train_loader = DataLoader(train_dataset, batch_size=8, pin_memory=True, shuffle=True, sampler=None, num_workers=0)

    # Load and prepare the validation data
    val_dataset = LFWDataset(download=False, base_folder='lfw_dataset',
                             split_name="val", transforms=None)  # Ensure you have a validation split
    val_loader = DataLoader(val_dataset, batch_size=8, pin_memory=True, shuffle=False, sampler=None, num_workers=0)

    encoder_channels = [3, 64, 128, 256, 512, 1024]
    decoder_depths = [512, 256, 128, 64]
    num_classes = 3  # Number of classes in the segmentation problem

    # Initialize the U-Net model with 3 input channels and 3 classes
    model = UNet(encoder_channels, decoder_depths, num_classes)
    model = model.to(device)  # Move the model to the specified device (CPU)

    # Set up the optimizer. Using Adam optimizer with a learning rate of 0.0001
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

    # Define the loss function - CrossEntropyLoss for multi-class segmentation
    criterion = nn.CrossEntropyLoss()

    # Number of epochs to train the model
    num_epochs = 70

    # Training loop
    for epoch in range(num_epochs):
        train(train_loader, val_loader, model, optimizer, criterion, epoch, num_epochs, device)
