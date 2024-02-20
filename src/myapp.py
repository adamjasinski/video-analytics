import streamlit as st
import av
import cv2
import numpy as np
import io

def convert_to_bw(byte_array):
    # Convert byte array to numpy array
    np_array = np.frombuffer(byte_array, np.uint8)

    # Open video file
    in_container = av.open(io.BytesIO(np_array))

    # Create output container in memory
    output = io.BytesIO()
    out_container = av.open(output, mode='w', format='mp4')

    # Create stream
    out_stream = out_container.add_stream('mpeg4')

    for frame in in_container.decode(video=0):
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(np.array(frame.to_image()), cv2.COLOR_BGR2GRAY)

        # Convert grayscale frame back to image
        img = av.VideoFrame.from_ndarray(gray_frame, format='gray')

        # Write frame to output container
        packet = out_stream.encode(img)
        out_container.mux(packet)

    # Close the output container
    out_container.close()

    # Get byte array from output
    bw_byte_array = output.getvalue()
    return bw_byte_array

st.header("Video converter")
uploaded_video = st.file_uploader("Select a video", type=["mp4", "avi"])

if uploaded_video is not None:
    # To read file as bytes:
    bytes_data = uploaded_video.getvalue()

    #converted = convert_to_bw(bytes_data)
    #assert converted is not None
    converted = bytes_data
    st.video(data=converted)