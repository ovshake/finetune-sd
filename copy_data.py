import json
import shutil
import os

# Destination directory
dest_dir = "images-test/train"

# Create the directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

with open("/home/ubuntu/projects/finetune-sd/images/train/metadata.jsonl", 'r') as f:
    for line in f:
        data = json.loads(line)
        file_name = data['file_name']

        # Copy the file to the destination directory
        shutil.copy(file_name, dest_dir)
