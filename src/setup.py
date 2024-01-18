import logging
import os
from zipfile import ZipFile

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def download_file(url: str, filename: str):
    """
    :param url: Link to download file from.
    :param filename: filename to save file to.
    """
    os.makedirs(os.path.dirname(filename) or '.'
                , exist_ok=True)

    response = requests.get(url, stream=True)
    file_size = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=file_size, unit="iB", unit_scale=True)
    block_size = 2048

    with open(filename, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
        progress_bar.close()

        if file_size != 0 and progress_bar.n != file_size:
            logging.info("ERROR, something went wrong")


def setup_dataset(folder_path: str, zip_file_path: str, url: str):
    """
    :param folder_path: the desired folder path of the dataset.
    :param zip_file_path: the zip file path of the downloaded dataset to use if the folder path is not present,
    :param url: the URL  link to download the file from, if the zip file path is not present.
    """
    if os.path.exists(folder_path):
        logging.info(f"Folder {folder_path} already exists. Using this directory as the dataset.")
    else:

        if zip_file_path[-4:] != ".zip":
            logging.info("Path: " + zip_file_path[-4:] + ". Adding '.zip' to path: ")
            zip_file_path = zip_file_path + ".zip"
        if not os.path.exists(zip_file_path):
            logging.info(f"Downloading dataset from {url} to {zip_file_path}.")
            download_file(url, zip_file_path)

        logging.info(f"Extracting {zip_file_path} to {folder_path}.")
        with ZipFile(zip_file_path, "r") as zip_ref:
            zip_ref.extractall(folder_path)
        os.remove(zip_file_path)
        logging.info("Dataset setup complete.")


if __name__ == "__main__":
    dataset_folder_path = "data"
    dataset_zip_file_path = "data"
    dataset_url = "https://lilablobssc.blob.core.windows.net/conservationdrones/v01/conservation_drones_train_real.zip"
    setup_dataset(dataset_folder_path, dataset_zip_file_path, dataset_url)
