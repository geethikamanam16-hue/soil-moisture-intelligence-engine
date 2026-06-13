from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

# ==========================================
# AUTH USING SERVICE ACCOUNT
# ==========================================

gauth = GoogleAuth()

scope = ['https://www.googleapis.com/auth/drive']

gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
    'cloud/service_account.json',
    scope
)

drive = GoogleDrive(gauth)

print("Service Account Authenticated Successfully.\n")

# ==========================================
# YOUR GOOGLE DRIVE FOLDER ID
# ==========================================

FOLDER_ID = "1a0IqDu4aE_wHRU9M4DrVyDxxuRL0YG99"

# ==========================================
# GET FILES FROM DRIVE FOLDER
# ==========================================

file_list = drive.ListFile({
    'q': f"'{FOLDER_ID}' in parents and trashed=false"
}).GetList()

data = []

# ==========================================
# PROCESS FILES
# ==========================================

for file in file_list:

    filename = file['title']

    print(f"Found: {filename}")

    data.append({
        "filename": filename,
        "gdrive_id": file['id']
    })

# ==========================================
# SAVE METADATA CSV
# ==========================================

df = pd.DataFrame(data)

df.to_csv("drive_file_index.csv", index=False)

print("\nMetadata CSV created successfully.")
print(f"Total files found: {len(df)}")