from diffusers import StableDiffusionPipeline
import os
import json
import shutil
import torch
original_model_id = "runwayml/stable-diffusion-v1-5"
finetuned_model_id = "/home/ubuntu/projects/diffusers/examples/text_to_image/unsplash-lite-v1"
def make_pipeline(model_id):
    """
    Create a diffusion pipeline from a pretrained model.

    Args:
    model_id (str): The identifier of the pretrained model.

    Returns:
    pipe (StableDiffusionPipeline): The diffusion pipeline created from the pretrained model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
    pipe = pipe.to(device)
    return pipe

def generate_image(pipe, text):
    """
    Generate an image from a given text using the provided diffusion pipeline.

    Args:
    pipe (StableDiffusionPipeline): The diffusion pipeline to use for image generation.
    text (str): The text to generate the image from.

    Returns:
    image (PIL.Image): The generated image.
    """
    image = pipe(text).images[0]
    return image


def compare(original_model_id, finetuned_model_id, metadata_path):
    """
    Compares the output of the original model and the finetuned model using the metadata from a file.

    Args:
    original_model_id (str): The identifier of the original model.
    finetuned_model_id (str): The identifier of the finetuned model.
    metadata_path (str): The path to the metadata file.
    """
    with open(metadata_path, 'r') as f:
        metadata = json.loads(f)
        text = metadata['text']


def copy_data(metadata_path):
    """
    Copies image files referenced in a metadata file to a specific directory.

    Args:
    metadata_path (str): The path to the metadata file.
    """
    os.makedirs("test/gt_images", exist_ok=True)
    with open(metadata_path, 'r') as f:
        for line in f.readlines():
            metadata = json.loads(line)
            image_path = metadata['file_name']
            shutil.copy(image_path, "test/gt_images")
            print(f"DONE COPYING {image_path} to test/gt_images")


if __name__ == '__main__':
    os.makedirs("test/sd1-5", exist_ok=True)
    os.makedirs("test/finetuned", exist_ok=True)
    original_pipeline = make_pipeline(original_model_id)
    finetuned_pipeline = make_pipeline(finetuned_model_id)
    metadata_path = "/home/ubuntu/projects/finetune-sd/metadata-test.jsonl"
    with open(metadata_path, 'r') as f:
        for line in f.readlines():
            metadata = json.loads(line)
            text = metadata['text']
            sd1_5_image = generate_image(original_pipeline, text)
            sd1_5_image.save(f"test/sd1-5/{metadata['file_name'].split('/')[-1]}")
            finetuned_image = generate_image(finetuned_pipeline, text)
            finetuned_image.save(f"test/finetuned/{metadata['file_name'].split('/')[-1]}")
            print(f"DONE GENERATING IMAGES FOR {metadata['file_name']}")
