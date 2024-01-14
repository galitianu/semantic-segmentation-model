import torch

def calculate_segmentation_metrics(predicted, target, C):
    """
    Calculate all segmentation metrics: MPA, mIoU, and FWIoU.

    :param predicted: Tensor of predicted segmentation maps (batch_size, C, H, W)
    :param target: Tensor of ground truth segmentation maps (batch_size, H, W)
    :param C: Number of classes
    :return: A dictionary containing Mean Pixel Accuracy, Mean IoU, and Frequency Weighted IoU
    """
    # Calculate n_ij matrix
    predicted_classes = predicted.argmax(dim=1)
    target_classes = target.argmax(dim=1)
    n_ij = torch.zeros((C, C), dtype=torch.int64)
    for i in range(C):
        for j in range(C):
            n_ij[i, j] = torch.sum((target_classes == i) & (predicted_classes == j))
    # Calculate t_i
    t_i = n_ij.sum(dim=1)

    # Extract diagonal elements (n_ii) which represent correctly classified pixels for each class
    n_ii = torch.diag(n_ij)

    # Mean Pixel Accuracy (MPA)
    mpa = (n_ii.float() / t_i.float().clamp(min=1)).mean().item()

    # Mean Intersection over Union (mIoU)
    union = t_i + n_ij.sum(dim=0) - n_ii
    iou = n_ii.float() / union.float().clamp(min=1)
    mIoU = iou.mean().item()

    # Frequency Weighted Intersection over Union (FWIoU)
    total_pixels = t_i.sum()
    fwIoU = (t_i.float() / total_pixels).dot(iou).item()

    return mpa, mIoU, fwIoU