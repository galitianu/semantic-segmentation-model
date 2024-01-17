import io

import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image, ImageFilter
from torchvision import transforms

from log import class_to_rgb
from model.UNet import UNet


def blur_background(input_image):
    # Load ScriptModule from io.BytesIO object
    with open(rf'model.pt', 'rb') as f:
        buffer = io.BytesIO(f.read())

    encoder_channels = [3, 64, 128, 256, 512, 1024]
    decoder_depths = [512, 256, 128, 64]
    num_classes = 3  # Number of classes in the segmentation problem

    # Initialize the U-Net model with 3 input channels and 3 classes
    model = UNet(encoder_channels, decoder_depths, num_classes)
    model.load_state_dict(torch.load(buffer, map_location=torch.device('cpu')))

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(286),
        transforms.CenterCrop(286),
    ])

    input_tensor = preprocess(input_image)
    original_image_copy = torch.clone(input_tensor)

    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        # run the model
        output = model(input_batch)
        output = output.squeeze(0)
        output = output.permute(1, 2, 0)

    prediction = torch.argmax(output, dim=2)
    pred_np = prediction.cpu().numpy()
    pred_rgb = class_to_rgb(pred_np)

    # Convert pred_rgb to a mask based on its blue channel
    blue_channel = pred_rgb[:, :, 2]
    mask = blue_channel > 0  # Adjust this threshold as needed

    # Convert the mask to a PIL Image for easier processing
    mask_image = Image.fromarray(mask.astype('uint8') * 255)

    # Convert the processed image for blurring

    # Convert the original image copy to PIL for final composite
    original_pil = T.ToPILImage()(original_image_copy)

    # Ensure the mask is the same size as the original image
    original_pil = original_pil.resize(mask_image.size)
    blurred_image = original_pil.filter(ImageFilter.GaussianBlur(radius=5))
    print(blurred_image.size, original_pil.size, mask_image.size)
    # Combine the blurred and original images using the mask
    final_image = Image.composite(blurred_image, original_pil, mask_image)

    return final_image


if __name__ == '__main__':
    webcam = gr.Image(height=286, width=286, sources=["webcam"], streaming=True)
    webapp = gr.Interface(fn=blur_background, inputs=webcam, outputs="image")

    webapp.launch()
