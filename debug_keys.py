import gspread
from google.oauth2.service_account import Credentials

creds = Credentials.from_service_account_file("credentials.json", scopes=[
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly"
])

client = gspread.authorize(creds)

sheet = client.open("SentimentFeedback").sheet1

records = sheet.get_all_records()

print("First record keys:", records[0].keys())
