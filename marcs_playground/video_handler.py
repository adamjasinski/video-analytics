import os
import io
from typing import Callable
import cv2
import uuid
import pandas as pd


class VideoHandler():

    def __init__(self, video: bytes):
        """ Constructor """
        self.byte_video = io.BytesIO(video.read())
        self.video_path = str(uuid.uuid4()) + ".mp4"

        #write file to given video_path
        with open(self.video_path, 'wb') as out:
            out.write(self.byte_video.read())
        
    def get_video_path(self):
        return self.video_path

    def get_video_stats(self) -> tuple:
        """
        Retrieve statistics about a video file.

        Returns:
        - tuple: A tuple containing video statistics in the following format:
            (fps (int), width (int), height (int), total_frames (int))
        """
        vf = cv2.VideoCapture(self.video_path)

        fps = int(vf.get(cv2.CAP_PROP_FPS))
        width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))

        return (fps, width, height, total_frames)
    
    def track(self,progressbar_callback: Callable) -> pd.DataFrame:
        """ """
        progressbar_callback(50)
        return None


    
    def __del__(self):
        """
        Remove a video file.
        """
        os.remove(self.video_path)
