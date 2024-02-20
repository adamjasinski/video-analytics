import ultralytics
import os
import torch
from video_helper import get_video_properties

event_counter = 0
frames = 0

def add_callback_for(model, evt):
    def my_callback(x):
        global event_counter
        global frames

        if evt == "on_predict_batch_start":
            event_counter += 1
            print(f"Callback on_predict_batch_start number {event_counter}/{frames}")
        else:
            print(f"Callback for event {evt} called; args: {x}")

    model.add_callback(evt, my_callback)


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
    add_callback_for(model, evt)
# model.add_callback("on_predict_postprocess_end", my_callback)
# model.add_callback("on_predict_start", my_callback)
# model.add_callback("on_predict_end", my_callback)
# model.add_callback("on_predict_batch_start", my_callback)
# model.add_callback("on_predict_batch_end", my_callback)

print(model.device)
# model.device = device
# model.overrides["device"] = device

#benchmark_results = model.benchmark()
#print(benchmark_results)


#results = model.track(source="./test/data/cars-downsampled.mp4", conf=0.3, iou=0.5, show=True, kwargs=dict(device=device))
#kws = dict(source="./test/data/cars-downsampled.mp4", conf=0.3, iou=0.5, show=True, device=device)
#results = model.track(**kws)

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