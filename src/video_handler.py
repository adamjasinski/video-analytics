import os
import io
from typing import Callable
import cv2
import uuid
import pandas as pd
import tempfile
import ultralytics
import torch
from yolo_helper import make_callback_adapter_with_counter, convert_tracking_results_to_pandas

class VideoHandler():

    def __init__(self, video: bytes):
        """ Constructor """
        self.byte_video = io.BytesIO(video.read())
        self.temp_id = str(uuid.uuid4())
        self.video_path = self.temp_id + ".mp4"

        #write file to given video_path
        with open(self.video_path, 'wb') as out:
            out.write(self.byte_video.read())
        
    def get_video_path(self):
        return self.video_path
    
    # def get_tracked_video_file(self)

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
    
    def track(self,progressbar_callback: Callable) -> tuple[pd.DataFrame, str]:
        """
        Perform object tracking on the video with YOLOv8.
        Args:
            progressbar_callback (Callable[int]): a callback accepting 1 argument (frame number)
        Return:
            DataFrame with tracking results
            Processed video path
        """
        pretrained_model = "yolov8n.pt"
        model = ultralytics.YOLO(pretrained_model, verbose=True)
        yolo_progress_reporting_event = "on_predict_batch_start"
        progress_callback_wrapped = make_callback_adapter_with_counter(yolo_progress_reporting_event, 
                                                                       lambda _,counter: progressbar_callback(counter))
        model.add_callback(yolo_progress_reporting_event, progress_callback_wrapped)

        device = 0 if torch.cuda.is_available() else 'cpu' 
        video_output = None
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"Using tmpdir: {tmpdir}")
            tracking_results = model.track(source=self.video_path, conf=0.2, iou=0.6, show=False, device=device, stream=True, save=True, save_dir=tmpdir)
            results_df = convert_tracking_results_to_pandas(tracking_results)
            # NB - workaround for the bug in Ultralytics that ignores the path passed to save_dir
            #processed_video_path = os.path.join(tmpdir, "runs/detect/track", self.temp_id + ".avi")
            processed_video_path = os.path.join("runs/detect/track", self.temp_id + ".avi")
            converted_video_path = os.path.join("runs/detect/track", self.temp_id + ".mp4")
            # TODO - huge security hole! replace with encoding via PyAV library
            os.system(f"ffmpeg -y -i {processed_video_path} -vcodec libx264 {converted_video_path}")
        return results_df, converted_video_path
    
    def __del__(self):
        """
        Remove a video file.
        """
        os.remove(self.video_path)
