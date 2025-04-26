import tonic
import tonic.transforms as TT
import torch

SENSOR_SIZE = (1280, 720, 2)
transform = TT.ToFrame(sensor_size=SENSOR_SIZE, time_window=10000)

def load_from_aedat(file_path):
    return tonic.io.read_aedat4(file_path)

def transform_events_to_frames(events):
    frames = transform(events)
    frames = torch.tensor(frames.transpose(1, 0, 2, 3), dtype=torch.float32)
    return frames
