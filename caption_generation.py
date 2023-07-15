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
# blip2_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b-coco")
# blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b-coco", device_map="auto", torch_dtype=torch.float16)
blip_processor_large = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model_large = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)


def read_image(image_url):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        return None


def generate_caption(processor, model, image, tokenizer=None, use_float_16=False):
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
    caption_blip_large = generate_caption(blip_processor_large, blip_model_large, image)
    return caption_blip_large


def fetch_data():
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
    # image = read_image(image_url)
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
    with open("metadata-test.jsonl", 'w') as f:
        for line in caption_dicts:
            f.write(json.dumps({"file_name": line["image_url"], "text": line["text"]}) + '\n')

if __name__ == '__main__':
    # Parse command-line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('min_index', type=int, help='Minimum index')
    # parser.add_argument('max_index', type=int, help='Maximum index')
    # args = parser.parse_args()
    image_paths = glob.glob("/home/ubuntu/projects/finetune-sd/images/train/*.jpg")
    image_paths = image_paths[-10:]
    caption_dicts = []
    # Slice the image_urls list according to the provided indices
    # image_urls = image_urls[args.min_index:args.max_index]
    for image_url in tqdm(image_paths):
        caption_dict = process_captions(image_url)
        if caption_dict:
            caption_dicts.append(caption_dict)

    save_caption_dicts(caption_dicts)



