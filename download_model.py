import os

import gdown


def download_model() -> None:
    model_url = (
        "https://drive.google.com/drive/folders/1QEAV4yRVRaSfyvtmh8KntPyJeel6uMlz"
    )
    destination_folder = "./triton/my_bert_mc/"
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)
    gdown.download_folder(url=model_url, output=destination_folder, quiet=False)

if __name__ == "__main__":
    download_model()
