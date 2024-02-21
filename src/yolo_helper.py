import ultralytics
import os
import torch
from video_helper import get_video_properties
import pandas as pd
import numpy as np

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

def _convert_single_tracking_result(frame_no, boxes_result:ultralytics.engine.results.Boxes):
    box = boxes_result.boxes # sic!
    int_vectorized = np.vectorize(np.int_, otypes=[int])
    if box is not None:
        class_ids = int_vectorized(box.cls.cpu().numpy())
        observation_count = len(class_ids)
        class_id_to_name = lambda id: boxes_result.names[int(id)]
        class_names = list(map(class_id_to_name, class_ids))
        ids = int_vectorized(box.id.cpu()) if box.id is not None else np.zeros(shape=observation_count, dtype='int')
        xywh = box.xywh.cpu()
        xs = xywh[:, 0]
        ys = xywh[:, 1]
        ws = xywh[:, 2]
        hs = xywh[:, 3]
        frame_nos = np.repeat(a=frame_no, repeats=observation_count)
        data = dict(frame_no=frame_nos, class_id=class_ids, class_name=class_names, id=ids, x=xs, y=ys, w=ws, h=hs)
        df = pd.DataFrame(data=data)
        return df
    else:
        return pd.DataFrame(columns=['frame_no','class_id', 'class_name', 'id', 'x', 'y', 'w', 'h'])

def convert_tracking_results_to_pandas(tracking_results):
    """
    Convert YOLOv8 tracking output to a Pandas DataFrame.
    The DataFrame contains the following columns:
        - frame_no:int - frame number
        - class_id:int - class identifier
        - class_name:str - class name of the tracked object
        - id:int - identifier of the tracked object
        - x:int - coordinates of the bounding boxes
        - y:int
        - w:int
        - h::int
    """
    dfs = [] # Will contain 1 data frame per video frame
    for i, tr in enumerate(tracking_results):
        df = _convert_single_tracking_result(i, tr)
        dfs.append(df)

    return pd.concat(dfs)
    # counter = 0
    # data_frames_per_frame = []
    # for box in boxes:
    #     class_ids = box.cls.numpy()
    #     ids = box.id.numpy()
    #     data = dict(class_id=class_ids, id=ids)
    #     df = pd.DataFrame(data=data)
    #     data_frames_per_frame.append(df)
    #     counter += 1
    #     if counter >= 20:
    #         break
    # return pd.concat(data_frames_per_frame, axis=0)

if __name__ == '__main__':
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.cuda.set_device(0) # Set to your desired GPU number

    print(f"Current path: {os.getcwd()}")

    #model = ultralytics.YOLO('yolov8n.yaml').load('yolov8n.pt')

    model = ultralytics.YOLO('yolov8n.pt', verbose=True)
    events_to_register = [
        "on_predict_postprocess_end", 
        "on_predict_start",
        "on_predict_end",
        "on_predict_end",
        "on_predict_batch_start",
        "on_predict_batch_end"]
    for evt in events_to_register:
        #add_callback_for(model, evt)
        callback_with_counter = make_callback_adapter_with_counter(evt, 
                                                                lambda evt, counter: 
                                                                    print(f"Callback for event {evt} called; counter:{counter}"))
        model.add_callback(evt, callback_with_counter)

    print(model.device)
    # model.device = device
    # model.overrides["device"] = device

    #benchmark_results = model.benchmark()
    #print(benchmark_results)

    source_video = "./test/data/cars-downsampled.mp4"
    #source_video = "/home/adam/Projects/DSR/final_project/data/cutted/machine_vs_condors/machine_vs_condors_pool_001.mp4"

    props = get_video_properties(source=source_video)
    frames = props["frames"]
    print(f"Frames: {frames}")
    results = model.track(source=source_video, conf=0.2, iou=0.6, show=False, device=0, stream=True)
    assert results is not None
    print(f"Results type: {type(results)}")

    combined_df = convert_tracking_results_to_pandas(results)
    # counter = 0
    # dfs = []
    # for res in results:
    #     #print(res.names) - all classes explained
    #     print(res)
    #     print("Box")
    #     #print(res.boxes[0])
    #     df = convert_tracking_result_to_pandas(res)
    #     #print(df)
    #     dfs.append(df)
    #     counter += 1
    #     # if counter >= 20:
    #     #     break
    
    # combined_df = pd.concat(dfs)
    print(combined_df.head(30))
    print(f"Total length: {len(combined_df)}")
    #print(combined_df[combined_df[id > 1.0]].head(30))

    # print(f"Result type: {type(results[0])}")
    # print(results[0])
    # print("Boxes")
    # print(results[0].boxes[0])
        
