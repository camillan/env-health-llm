from dotenv import load_dotenv
import os

load_dotenv()
print("HF_TOKEN =", os.getenv("HF_API_TOKEN"))
