import torch
import wandb


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
