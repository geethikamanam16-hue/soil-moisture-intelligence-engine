from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

import os
import zipfile

# ==========================================
# AUTH
# ==========================================

gauth = GoogleAuth()

scope = ['https://www.googleapis.com/auth/drive']

gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
    'cloud/service_account.json',
    scope
)

drive = GoogleDrive(gauth)

print("Authenticated Successfully.\n")

# ==========================================
# GOOGLE DRIVE FOLDER ID
# ==========================================

FOLDER_ID = "1Wz6NvuB12Aa0PM5s6v4IYgEtufnb-YAF"

# ==========================================
# LOCAL CACHE
# ==========================================

CACHE_FOLDER = r"cache\zarr"

os.makedirs(CACHE_FOLDER, exist_ok=True)

# ==========================================
# GET FILE LIST
# ==========================================

file_list = drive.ListFile({
    'q': f"'{FOLDER_ID}' in parents and trashed=false"
}).GetList()

print("Found files:\n")

# ==========================================
# DOWNLOAD ALL
# ==========================================

for file in file_list:

    filename = file['title']

    print(filename)

    # only zips
    if not filename.endswith(".zip"):
        continue

    zip_path = os.path.join(
        CACHE_FOLDER,
        filename
    )

    # skip if already exists
    if os.path.exists(zip_path):
        print(f"Skipping existing ZIP: {filename}")
        continue

    print(f"\nDownloading {filename}...")

    file.GetContentFile(zip_path)

    print("Download complete.")

    # ==========================================
    # EXTRACT
    # ==========================================

    print("Extracting ZIP...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(CACHE_FOLDER)

    print(f"✔ Extracted: {filename}\n")

print("\nALL DOWNLOADS COMPLETE.")