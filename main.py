from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import uvicorn
import tempfile
import os
import subprocess
import tonic
import utils
import json
import logging
import cv2
import lightning_model
import torch

BASE_LLM_URL = "http://localhost:1234/v1"
LLM_API_KEY = "lm-studio"
LLM_MODEL = "deepseek-r1-distill-qwen-7b"
# LLM_MODEL = "mistral-7b-instruct-v0.3"

with open('labels100.json', 'r') as f:
    labels = json.load(f)
    structured_labels = {x['target']: x['label'] for x in labels}

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/upload-video")
async def upload_video(video: UploadFile = File(...)):
    try:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as temp_file:
            content = await video.read()
            logger.debug(f"File size: {len(content)}")
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
            temp_file_path = temp_file.name

        # Convert video to events with v2e

        # Define output path
        output_dir = tempfile.mkdtemp()
        output_aedat_file_path = os.path.join(output_dir, "data.aedat4")
        output_dvs_video_path = os.path.join(output_dir, "dvs-video.avi")
        logger.debug("Output path created")

        file_size = os.path.getsize(temp_file_path)
        logger.debug(f"Temp file size on disk: {file_size} bytes")

        cap = cv2.VideoCapture(temp_file_path)
        logger.debug(f"frame count: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        logger.debug(f"frame rate: {cap.get(cv2.CAP_PROP_FPS)}")

        v2e_command = [
            "cmd.exe", "/C",
            f"call conda activate venv && "
            f'v2e -i {temp_file_path} -o {output_dir} --input_frame_rate 30 --dvs346 --no_preview --disable_slomo --dvs_aedat4 data.aedat --overwrite'
        ]

        logger.debug("Conda command set up")
        subprocess.run(v2e_command, check=True)
        logger.debug("Command executed")

        # Convert avi into mp4
        output_mp4_path = os.path.join(output_dir, "dvs-video.mp4")
        ffmpeg_command = [
            "ffmpeg",
            "-i", output_dvs_video_path,
            "-vcodec", "libx264",
            "-acodec", "aac",
            output_mp4_path,
            "-y"  # Overwrite without asking
        ]
        subprocess.run(ffmpeg_command, check=True)

        streaming_response = StreamingResponse(
            open(output_mp4_path, 'rb'),
            media_type='video/mp4'
        )

        dvs_dir_data = {
            "dir": output_dir,
            "data": output_aedat_file_path,
            "mp4_video": output_mp4_path,
            "dvs_video": output_dvs_video_path
        }
        with open("temp_dvs_dir.json", "w") as f:
            json.dump(dvs_dir_data, f)

        return streaming_response

    except Exception as e:
        return {"error": str(e)}
    
@app.get("/get-response")
async def get_response():
    try:
        with open("temp_dvs_dir.json", 'r') as f:
            dvs_dir_data = json.load(f)

        # Load the model
        model = lightning_model.get_model()
        model.eval()
        model.freeze()

        # Load and transform the data
        aedat_file_path = dvs_dir_data["data"]
        dvs_data = utils.load_from_aedat(aedat_file_path)
        frames = utils.transform_events_to_frames(dvs_data)
        frames = frames.unsqueeze(0)

        logger.debug("prediction started")
        preds = model.model(frames)["probs"]
        logger.debug("prediction ended")
        predicted_label = torch.argmax(torch.squeeze(preds), dim=0)
        predicted_label = str(predicted_label.item() + 1)

        if predicted_label not in structured_labels.keys():
            predicted_word = 'undefined'
        else:
            predicted_word = structured_labels[predicted_label]

        logger.debug(f'Predicted label: {predicted_label}')
        logger.debug(f'Predicted word: {predicted_word}')

        llm = ChatOpenAI(
            base_url=BASE_LLM_URL,
            api_key=LLM_API_KEY,
            temperature=0.9,
            model=LLM_MODEL
        )

        prompt = f"""
        A user has uploaded a video of themselves signing a word in the Word-Level American Sign Language.
        The pretrained CNN model predicted the word to be: {predicted_word}.

        If the word is undefined, please give the following response: "I wasn't able to recognize the word."

        If not, please start the response like this: "Your word was {predicted_word}". After this, give a more detailed explanation
        about the history of the sign and the etimology of the word in the context of the Word-Level American Sign Language.  
        """

        response = llm([HumanMessage(content=prompt)])

        message = response.content.split("</think>")[1]
        return message

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
