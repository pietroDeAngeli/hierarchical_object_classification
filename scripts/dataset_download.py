# This script downloads the validation dataset for the iNaturalist 2021 competition.

import requests
import os
from tqdm import tqdm

def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    filename = url.split('/')[-1]
    file_path = os.path.join(dest_folder, filename)

    print(f"Downloading {url} to {file_path}")
    
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(file_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    else:
        print(f"Download complete: {file_path}")

if __name__ == "__main__":
    # Data download
    url = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz"
    download_file(url, "iNaturalist2021_val")

    # Metadata download
    url = "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz"
    download_file(url, "iNaturalist2021_val")
