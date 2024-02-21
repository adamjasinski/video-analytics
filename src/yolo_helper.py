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

    #source_video = "./test/data/cars-downsampled.mp4"
    source_video = "/home/adam/Projects/DSR/final_project/data/cutted/machine_vs_condors/machine_vs_condors_pool_001.mp4"

    props = get_video_properties(source=source_video)
    frames = props["frames"]
    print(f"Frames: {frames}")
    results = model.track(source=source_video, conf=0.2, iou=0.6, show=False, device=0, stream=True)
    assert results is not None
    print(f"Results type: {type(results)}")

    counter = 0
    for res in results:
        #print(res.names) - all classes explained
        print(res)
        print("Box")
        print(res.boxes[0])
        counter += 1
        if counter >= 20:
            break
    # print(f"Result type: {type(results[0])}")
    # print(results[0])
    # print("Boxes")
    # print(results[0].boxes[0])
        
