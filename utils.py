import torch
import cv2
import numpy as np
import tempfile
import os
from typing import List
import json
from datetime import datetime

def video_to_tensor(video_path: str, max_frames: int = None) -> torch.Tensor:
    """
    Convert video to PyTorch tensor.
    Returns tensor of shape (T, C, H, W) where:
    T = number of frames
    C = channels (3 for RGB)
    H = height
    W = width
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Convert to tensor format (H, W, C) -> (C, H, W)
        frame = np.transpose(frame, (2, 0, 1))
        frames.append(frame)
        
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    
    if not frames:
        raise ValueError("No frames were read from the video")
    
    # Stack frames into a single tensor
    video_tensor = torch.stack([torch.from_numpy(frame) for frame in frames])
    return video_tensor

def get_tensor_info(tensor: torch.Tensor) -> dict:
    """
    Get basic information about a tensor.
    """
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min_value": float(tensor.min()),
        "max_value": float(tensor.max()),
        "mean_value": float(tensor.mean()),
    }

def log_tensor_info(filename: str, tensor_info: dict, log_dir: str = "logs") -> str:
    """
    Log tensor information to a JSON file.
    
    Args:
        filename: Original video filename
        tensor_info: Dictionary containing tensor information
        log_dir: Directory to store log files (default: "logs")
    
    Returns:
        str: Path to the log file
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log entry with timestamp
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "tensor_info": tensor_info
    }
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"tensor_log_{timestamp}.json"
    log_path = os.path.join(log_dir, log_filename)
    
    # Write to log file
    with open(log_path, 'w') as f:
        json.dump(log_entry, f, indent=4)
    
    return log_path 