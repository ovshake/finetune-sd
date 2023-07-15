from transformers import AutoProcessor, AutoTokenizer, BlipForConditionalGeneration
import torch
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
import glob
from joblib import Parallel, delayed
import requests
import os
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import json
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
blip_processor_large = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model_large = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)


def read_image(image_url):
    """
    Fetches an image from a URL and opens it as a PIL Image object.

    Args:
    image_url (str): URL of the image to be fetched and opened.

    Returns:
    img (PIL.Image.Image or None): Opened image object, or None if the image could not be opened.
    """
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        return None


def generate_caption(processor, model, image, tokenizer=None, use_float_16=False):
    """
    Generates a caption for a given image using a specified processor and model.

    Args:
    processor (transformers.PreTrainedProcessor): Processor to preprocess the image.
    model (transformers.PreTrainedModel): Model to generate the caption.
    image (PIL.Image.Image): Image to be captioned.
    tokenizer (transformers.PreTrainedTokenizer, optional): Tokenizer to decode the caption.
    use_float_16 (bool, optional): Whether to use float16 for inputs, defaults to False.

    Returns:
    generated_caption (str): Generated caption for the image.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)

    if use_float_16:
        inputs = inputs.to(torch.float16)

    generated_ids = model.generate(pixel_values=inputs.pixel_values, num_beams=3, max_length=20, min_length=5)

    if tokenizer is not None:
        generated_caption = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    else:
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_caption



def generate_captions(image):
    """
    Generates a caption for a given image using the global 'blip_processor_large' and 'blip_model_large'.

    Args:
    image (PIL.Image.Image): Image to be captioned.

    Returns:
    caption_blip_large (str): Generated caption for the image.
    """
    caption_blip_large = generate_caption(blip_processor_large, blip_model_large, image)
    return caption_blip_large


def fetch_data():
    """
    Fetches data from files stored in the './data/photos' directory and combines them into a pandas DataFrame.

    Returns:
    datasets (dict): Dictionary containing a combined pandas DataFrame of all the fetched data.
    """
    path = './data/'
    documents = ['photos']
    datasets = {}

    for doc in documents:
        files = glob.glob(path + doc + ".tsv*")
        print(files)

        subsets = []
        for filename in files:
            df = pd.read_csv(filename, sep='\t', header=0)
            subsets.append(df)

        datasets[doc] = pd.concat(subsets, axis=0, ignore_index=True)

    return datasets

def process_captions(image_url):
    """
    Processes captions for an image at a given URL.

    Args:
    image_url (str): URL of the image to process.

    Returns:
    caption_dict (dict or None): Dictionary containing the image URL and its caption, or None if the image could not be opened.
    """
    image = Image.open(image_url)
    if not image:
        return None
    caption = generate_captions(image)
    caption_dict = {
        'image_url': image_url,
        'text': caption
    }
    return caption_dict

def save_caption_dicts(caption_dicts):
    """
    Saves a list of dictionaries containing image metadata into a JSONL file.

    Args:
    caption_dicts (list of dict): List of dictionaries containing image metadata to save.
    """
    with open("metadata-test.jsonl", 'w') as f:
        for line in caption_dicts:
            f.write(json.dumps({"file_name": line["image_url"], "text": line["text"]}) + '\n')

if __name__ == '__main__':
    image_paths = glob.glob("/home/ubuntu/projects/finetune-sd/images/train/*.jpg")
    image_paths = image_paths[-10:]
    caption_dicts = []
    for image_url in tqdm(image_paths):
        caption_dict = process_captions(image_url)
        if caption_dict:
            caption_dicts.append(caption_dict)

    save_caption_dicts(caption_dicts)



