from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
from utils import video_to_tensor, get_tensor_info, log_tensor_info

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/upload-video/")
async def upload_video(video: UploadFile = File(...)):
    try:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video.filename)[1]) as temp_file:
            content = await video.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Convert video to tensor
        video_tensor = video_to_tensor(temp_file_path)
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        # Get tensor information
        tensor_info = get_tensor_info(video_tensor)
        
        # Log tensor information
        log_path = log_tensor_info(video.filename, tensor_info)
        
        return {
            "filename": video.filename,
            "content_type": video.content_type,
            "message": "Video successfully converted to tensor",
            "tensor_info": tensor_info,
            "log_file": log_path
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)