import torch
import torch.nn as nn


def calculate_nij(predicted, target, C):
    """
    Calculate n_ij matrix.

    :param predicted: Tensor of predicted segmentation maps (batch_size, C, H, W)
    :param target: Tensor of ground truth segmentation maps (batch_size, H, W)
    :param C: Number of classes
    :return: n_ij matrix of shape (C, C)
    """
    # Convert predictions to class indices
    predicted_classes = predicted.argmax(dim=1)
    predicted_targets = target.argmax(dim=1)
    # Initialize n_ij matrix
    n_ij = torch.zeros((C, C), dtype=torch.int64)

    # Calculate n_ij
    for i in range(C):
        for j in range(C):
            n_ij[i, j] = torch.sum((predicted_targets == i) & (predicted_classes == j))

    return n_ij


def calculate_ti(n_ij):
    # Sum over columns to get total count of pixels per class in target
    t_i = n_ij.sum(dim=1)
    return t_i


def mean_pixel_accuracy(predicted, target, C):
    """
    Calculate the mean pixel accuracy.

    :param predicted: Tensor of predicted segmentation maps (batch_size, C, H, W)
    :param target: Tensor of ground truth segmentation maps (batch_size, H, W)
    :param C: Number of classes
    :return: Mean pixel accuracy
    """
    n_ij = calculate_nij(predicted, target, C)
    t_i = calculate_ti(n_ij)

    # Extract diagonal elements (n_ii) which represent correctly classified pixels for each class
    n_ii = torch.diag(n_ij)

    # Compute mean pixel accuracy
    pa = (n_ii.float() / t_i.float().clamp(min=1)).mean()

    return pa.item()


def mean_iou(predicted, target, C):
    """
    Calculate the mean Intersection over Union (mIoU).

    :param predicted: Tensor of predicted segmentation maps (batch_size, C, H, W)
    :param target: Tensor of ground truth segmentation maps (batch_size, H, W)
    :param C: Number of classes
    :return: Mean Intersection over Union
    """
    n_ij = calculate_nij(predicted, target, C)
    n_ii = torch.diag(n_ij)
    t_i = calculate_ti(n_ij)

    # Union for each class is the sum of the row and column for that class minus n_ii (to avoid double counting)
    union = t_i + n_ij.sum(dim=0) - n_ii

    # Calculate IoU for each class, and handle division by zero
    iou = n_ii.float() / union.float().clamp(min=1)

    # Compute mean IoU
    mIoU = iou.mean()

    return mIoU.item()


def frequency_weighted_iou(predicted, target, C):
    """
    Calculate the Frequency Weighted Intersection over Union (FWIoU).

    :param predicted: Tensor of predicted segmentation maps (batch_size, C, H, W)
    :param target: Tensor of ground truth segmentation maps (batch_size, H, W)
    :param C: Number of classes
    :return: Frequency Weighted Intersection over Union
    """
    n_ij = calculate_nij(predicted, target, C)
    n_ii = torch.diag(n_ij)
    t_i = calculate_ti(n_ij)

    # Union for each class
    union = t_i + n_ij.sum(dim=0) - n_ii

    # IoU for each class, and handle division by zero
    iou = n_ii.float() / union.float().clamp(min=1)

    # Total number of pixels across all classes
    total_pixels = t_i.sum()

    # Calculate FWIoU
    fwIoU = (t_i.float() / total_pixels).dot(iou)

    return fwIoU.item()
