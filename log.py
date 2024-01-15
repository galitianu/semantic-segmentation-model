import torch
import wandb
import numpy as np


def class_to_rgb(pred_np, num_classes=3):
    """
    Convert class predictions to RGB images.

    :param pred_np: 2D numpy array with class predictions.
    :param num_classes: Number of classes.
    :return: RGB image representation of the class predictions.
    """
    # Define a color for each class
    colors = {
        0: [255, 0, 0],  # Red
        1: [0, 255, 0],  # Green
        2: [0, 0, 255]  # Blue
    }

    # Create an empty array for the RGB image
    rgb_image = np.zeros((pred_np.shape[0], pred_np.shape[1], 3), dtype=np.uint8)

    # Assign the color to each pixel
    for cls in range(num_classes):
        rgb_image[pred_np == cls] = colors[cls]

    return rgb_image


def log_predictions(table, model, images, masks, device):
    model.eval()
    with torch.no_grad():
        # Forward pass
        predictions = model(images.to(device))

        for img, pred, mask in zip(images, predictions, masks):
            # Convert tensors to appropriate format for visualization
            # Permute to change from [channels, height, width] to [height, width, channels]
            img_np = img.cpu().permute(1, 2, 0).numpy()
            pred_np = pred.cpu().permute(1, 2, 0).numpy()
            mask_np = mask.cpu().permute(1, 2, 0).numpy()

            # Add data to table
            table.add_data(wandb.Image(img_np), wandb.Image(pred_np), wandb.Image(mask_np))
