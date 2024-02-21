import streamlit as st
#import av
import cv2
import numpy as np

from video_handler import VideoHandler


def main():
    
    st.title("Video Processing App")

    with st.sidebar:
        st.title("Configuration")
        st.subheader("Video uploader")

        uploaded_video = st.sidebar.file_uploader("Choose a video file", type=["mp4"])

        track_bar = st.sidebar.progress(0.0,text="Progressbar for tracking")

    if uploaded_video is not None:
        my_video = VideoHandler(uploaded_video)

        st.video(uploaded_video)

        fps, width, height, total_frames = my_video.get_video_stats()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(label="Frames per sec.", value=f"{fps}", delta=None)
        col2.metric(label="Width", value=f"{width} px", delta=None)
        col3.metric(label="Height", value=f"{height} px", delta=None)
        col4.metric(label="Frames", value=f"{total_frames}", delta=None)

        def update_progressbar(frame_counter):
            track_bar.progress(frame_counter/total_frames)

        tracking_results, video_output = my_video.track(progressbar_callback=update_progressbar)
        st.write(tracking_results)
        st.video(data=video_output)
        del my_video


if __name__ == "__main__":
    main()