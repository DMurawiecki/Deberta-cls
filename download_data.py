import os

import gdown


def download_data() -> None:
    folder_id_1 = "1ELxyBKdESWfiNmyX1AOsW6uph2h-XWMH"
    folder_id_2 = "1FxsuvrSfkPAyKePlwSCDgbIG8BSmkN5O"
    destination_folder_1 = "./"
    destination_folder_2 = "./bot"
    addresses = [
        (destination_folder_1, folder_id_1),
        (destination_folder_2, folder_id_2),
    ]

    for destination_folder, folder_id in addresses:
        os.makedirs(destination_folder, exist_ok=True)
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(url=folder_url, output=destination_folder, quiet=False)


def download_model_from_ckpt() -> None:
    model_url = (
        "https://drive.google.com/drive/folders/" "1Jouhh1dowuyycFIYVlMayZvkqYDPL82g"
    )
    destination_folder = "./"
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)
    gdown.download_folder(url=model_url, output=destination_folder, quiet=False)


if __name__ == "__main__":
    download_data()
    download_model_from_ckpt()
