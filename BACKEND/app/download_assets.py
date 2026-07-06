from pathlib import Path
import gdown

DRIVE_FOLDER = "https://drive.google.com/drive/folders/1ijOsUeSUfeSwHaG4zQpMKi0KWpAb7CSs"

LOCAL_DATA = Path("data")


def download_assets():

    if LOCAL_DATA.exists():
        print("Assets already downloaded.")
        return

    LOCAL_DATA.mkdir(parents=True, exist_ok=True)

    print("Downloading Kolrose assets...")

    gdown.download_folder(
        url=DRIVE_FOLDER,
        output=str(LOCAL_DATA),
        quiet=False,
        use_cookies=False,
    )

    print("Download complete.")