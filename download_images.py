import numpy as np
import pandas as pd
import glob
from joblib import Parallel, delayed
import requests
import os
from tqdm import tqdm
from PIL import Image
from io import BytesIO


def read_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

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

def save_image_from_url(image_url, image_id):
    # Specify the path of the directory where you want to save the image
    directory = "images"
    image_id = image_id.replace('-', '_')

    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

    if os.path.exists(os.path.join(directory, f"{image_id}.jpg")):
        return
    # Send a HTTP request to the specified URL and save the response from server in a response object called r
    try:
        r = requests.get(image_url)

    # Check if the request is successful
        if r.status_code == 200:
            # Open a binary file in write mode
            with open(os.path.join(directory, f"{image_id}.jpg"), 'wb') as f:
                # Write the content
                f.write(r.content)
        else:
            print(f"Unable to retrieve image. Server responded with status code: {r.status_code}")

    except:
        print(f"Unable to retrieve image. {image_id}")


if __name__ == '__main__':
    datasets = fetch_data()
    df = datasets['photos']
    image_urls = df['photo_image_url'].values
    image_ids = df['photo_id'].values
    for image_url, image_id in tqdm(zip(image_urls, image_ids)):
        save_image_from_url(image_url, image_id)
    # Parallel(n_jobs=-1)(delayed(save_image_from_url)(image_url, image_id) for image_url, image_id in tqdm(zip(image_urls, image_ids)))
