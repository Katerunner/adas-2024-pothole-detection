import os

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

roboflow_api_key = os.getenv("ROBOFLOW_API_KEY", "unauthorized")

rf = Roboflow(api_key=roboflow_api_key)
project = rf.workspace("crowsnest").project("pothole-u8cau")
version = project.version(1)
dataset = version.download("yolov8")
