import os

import gdown


def download_data() -> None:
    folder_id_1 = "1ELxyBKdESWfiNmyX1AOsW6uph2h-XWMH"
    folder_id_2 = "1FxsuvrSfkPAyKePlwSCDgbIG8BSmkN5O"
    destination_folder_1 = "./"
    destination_folder_2 = "./bot"
    adresses = [
        (destination_folder_1, folder_id_1),
        (destination_folder_2, folder_id_2),
    ]

    for destination_folder, folder_id in adresses:
        os.makedirs(destination_folder, exist_ok=True)
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(url=folder_url, output=destination_folder, quiet=False)


if __name__ == "__main__":
    download_data()
