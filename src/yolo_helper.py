import ultralytics
import os
import torch
from video_helper import get_video_properties

frames = 0

def make_callback_adapter_with_counter(event_name, callback):
    """
    Convert the callback function with 2 params to a callback format required by YOLO.
    Args:
        event_name: str: YOLO pipeline event name
        callback: Callable(str, int): a callback function accepting an 2 params: event_name and counter
    Return:
        A callback in the format required by YOLO.
    """
    event_counter = 0

    def yolo_callback(component):
        nonlocal event_counter
        event_counter += 1
        callback(event_name, event_counter)

    return yolo_callback