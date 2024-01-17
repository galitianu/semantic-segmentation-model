import io

import cv2
import gradio as gr
import numpy as np
import torch
from torchvision import transforms

from model.UNet import UNet


def blur_background(input_image):
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    # Load ScriptModule from io.BytesIO object
    with open(rf'model.pt', 'rb') as f:
        buffer = io.BytesIO(f.read())

    encoder_channels = [3, 64, 128, 256, 512, 1024]
    decoder_depths = [512, 256, 128, 64]
    num_classes = 3  # Number of classes in the segmentation problem

    # Initialize the U-Net model with 3 input channels and 3 classes
    model = UNet(encoder_channels, decoder_depths, num_classes)
    model.load_state_dict(torch.load(buffer))

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(286),
        transforms.CenterCrop(286),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        # run the model
        output = model(input_batch)

    # Use the last channel (channel index 2) of the model's output as the blur mask
    blur_mask = output[:, 2, :, :]  # Assuming the last channel corresponds to the mask

    # Convert the blur mask to a numpy array
    blur_mask = blur_mask.cpu().numpy()
    blur_mask = np.squeeze(blur_mask)  # Remove the batch dimension

    # Resize the blur mask to match the shape of input_image
    blur_mask = cv2.resize(blur_mask, (input_image.shape[1], input_image.shape[0]))

    # Apply Gaussian blur based on the mask
    blurred_input = cv2.GaussianBlur(input_image, (51, 51), 0)

    # Normalize pixel values to be between -1 and 1
    result = (input_image.astype(np.float32) / 255.0) * (1 - blur_mask[:, :, np.newaxis]) + (
                blurred_input.astype(np.float32) / 255.0) * blur_mask[:, :, np.newaxis]

    # Convert the result back to RGB format for Gradio
    result = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    return result


if __name__ == '__main__':
    webcam = gr.Image(height=640, width=480, sources=["upload"])
    webapp = gr.Interface(fn=blur_background, inputs=webcam, outputs="image")

    webapp.launch()
