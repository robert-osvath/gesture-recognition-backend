from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
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

with open('labels.json', 'r') as f:
    labels = json.load(f)

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
    logger.debug(video.headers)
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
            f'v2e -i {temp_file_path} -o {output_dir} --input_frame_rate 30 --output_height 720 --output_width 1280 --no_preview --disable_slomo --dvs_aedat4 data.aedat --overwrite'
        ]

        logger.debug("Conda command set up")
        subprocess.run(v2e_command, check=True)
        logger.debug("Command executed")

        # Load and convert
        events = utils.load_from_aedat(output_aedat_file_path)
        frames = utils.transform_events_to_frames(events)

        # Load model
        model = lightning_model.get_model()

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

        return streaming_response

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)