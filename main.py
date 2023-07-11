# main.py

import requests
import os
import warnings
import io

from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import replicate

def generate_and_upscale_image(text_prompt, clipdrop_api_key, stability_api_key, replicate_api_token):

    headers = {'x-api-key': clipdrop_api_key}
    body_params = {'prompt': (None, text_prompt, 'text/plain')}

    response = requests.post('https://clipdrop-api.co/text-to-image/v1',
                             files=body_params,
                             headers=headers)

    if response.status_code != 200:
        print(f"Request failed with status code {response.status_code}")
        return None, f"Request failed with status code {response.status_code}"

    with open('generated_image.png', 'wb') as f:
        f.write(response.content)

    os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
    os.environ['STABILITY_KEY'] = stability_api_key

    stability_api = client.StabilityInference(
        key=os.environ['STABILITY_KEY'],
        upscale_engine="esrgan-v1-x2plus",
        verbose=True,
    )

    max_pixels = 1048576
    img = Image.open('generated_image.png')
    width, height = img.size

    if width * height > max_pixels:
        scale_factor = (max_pixels / (width * height))**0.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        img = img.resize((new_width, new_height))

    answers = stability_api.upscale(init_image=img)

    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please submit a different image and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                upscaled_img = Image.open(io.BytesIO(artifact.binary))
                upscaled_img.save("upscaled_image.png")

    os.environ['REPLICATE_API_TOKEN'] = replicate_api_token
    Image.MAX_IMAGE_PIXELS = None

    with open("upscaled_image.png", "rb") as img_file:
        output = replicate.run(
            "tencentarc/gfpgan:9283608cc6b7be6b65a8e44983db012355fde4132009bf99d976b2f0896856a3",
            input={"img": img_file, "version": "v1.4", "scale": 16}
        )

    response = requests.get(output)
    if response.status_code != 200:
        return None, f"Failed to fetch upscaled image. Status code {response.status_code}"

    final_img = Image.open(io.BytesIO(response.content))
    final_img.save("gfpgan_upscaled_image.png")  # Save GFPGAN upscaled image

    # Scale to iPhone X format
    iphone_size = (2436, 1125)
    final_img = final_img.resize(iphone_size, Image.ANTIALIAS)

    # Save as WebP
    final_img.save("final_image.webp", "WEBP")

    # Open WebP and save as compressed JPG
    webp_img = Image.open("final_image.webp")
    webp_img.save("final_image_compressed.jpg", "JPEG", quality=95)

    return None, None

