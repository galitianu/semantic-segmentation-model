import cv2
import gradio as gr
import numpy as np
import torch
from torchvision import transforms


def blur_background(input_image):
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    # Generate a blank mask
    # TODO your code here: call a segmentation model to get predicted mask
    mask = np.zeros_like(input_image)

    model = torch.jit.load("scripted_resnet18.pt")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        # run the scripted model
        output = model(input_batch)
    print(output.size())
    # for demo purposes, we are going to create a random segmentation mask
    #  just a circular blob centered in the middle of the image
    center_x, center_y = mask.shape[1] // 2, mask.shape[0] // 2
    cv2.circle(mask, (center_x, center_y), 100, (255, 255, 255), -1)

    # Convert the mask to grayscale
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_gray = mask_gray[:, :, np.newaxis]

    # apply a strong Gaussian blur to the areas outside the mask
    blurred = cv2.GaussianBlur(input_image, (51, 51), 0)
    result = np.where(mask_gray, input_image, blurred)

    # Convert the result back to RGB format for Gradio
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result


webcam = gr.Image(height=640, width=480, sources=["webcam"], streaming=True)
webapp = gr.interface.Interface(fn=blur_background, inputs=webcam, outputs="image")

webapp.launch()
