# gs_test.py
import json, os
from google.oauth2.service_account import Credentials
import gspread

CREDS_PATH = "credentials.json"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly"
]

print("Using credentials file:", os.path.abspath(CREDS_PATH))

# load the JSON just to print it
with open(CREDS_PATH, "r") as f:
    creds_json = json.load(f)

print("client_email from JSON:", creds_json.get("client_email"))
print("project_id from JSON:", creds_json.get("project_id"))

# authorize
creds = Credentials.from_service_account_file(CREDS_PATH, scopes=SCOPES)
client = gspread.authorize(creds)

print("\nListing accessible spreadsheets...")
try:
    files = client.list_spreadsheet_files()
    if not files:
        print("No spreadsheets visible to this service account.")
    else:
        for f in files:
            print("-", f["name"], "| id:", f["id"])
except Exception as e:
    print("Error listing spreadsheets:", type(e).__name__, e)
