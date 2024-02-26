from typing import Union
import av
import cv2
import numpy as np
import io

def get_video_properties(source:Union[bytes|str]):
    """
    Return properties of a video.

    Parameters:
        source: Union[bytes|str]: file name or byte array
    Returns:
        A dictionary of video properties
    """
    if source is not None and isinstance(source, bytes):
        byte_array = source
    elif isinstance(source, str):
        with io.open(source, 'rb') as f:
            byte_array = f.read()

     # Create a in-memory bytes buffer
    buffer = av.open(io.BytesIO(byte_array))

    """Return properties of a video."""
    # Create a in-memory bytes buffer
    buffer = av.open(io.BytesIO(byte_array))

    # Get the video stream
    video_stream = next(s for s in buffer.streams if s.type == 'video')

    # Get the codec context
    codec_context = video_stream.codec_context

    # Now you can access properties
    width = codec_context.width
    height = codec_context.height
#    channels = codec_context.channels
    codec_name = codec_context.name
    frames = video_stream.frames
    duration_in_seconds = frames / (video_stream.average_rate)
    print(f'Width: {width}, Height: {height}, Codec: {codec_name}, Frames: {frames}, Duration: {duration_in_seconds}')

    return dict(height=height, width=width, codec=codec_name, duration=duration_in_seconds, frames=frames)

def encode_x264(byte_array):
    # Create an AVPacket from the byte array
    packet = av.Packet(byte_array)

    # Create a BytesIO object for the output
    output = io.BytesIO()

    # Create an AVStream for the output
    output_stream = av.open(output, 'w').add_stream('libx264', 24)

    # Open the codec context
    codec_context = av.CodecContext.create('libx264', 'w')
    codec_context.options = {
        'crf': 20,
        'preset': 'ultrafast'
    }

    # Decode the packet into a frame
    frame = packet.decode_one()

    # Encode the frame using libx264
    for packet in codec_context.encode(frame):
        output_stream.write(packet)

    # Close the stream
    output_stream.close()

    # Get the byte array from the BytesIO object
    byte_array = output.getvalue()

    return byte_array

def transcode_to_h264(byte_array: bytes) -> bytes:
    """
    Transcode the video to x264.
    Note: audio is skipped.
    """
    # Convert byte array to numpy array
    np_array = np.frombuffer(byte_array, np.uint8)

    # Open video file
    in_container = av.open(io.BytesIO(np_array))

    # Create output container in memory
    output = io.BytesIO()
    out_container = av.open(output, mode='w', format='mp4')

    # Create stream
    out_stream = out_container.add_stream('libx264', 24)

    # Open the codec context
    codec_context = av.CodecContext.create('libx264', 'w')
    #codec_context.pix_fmt = 'yuv420p'
    codec_context.options = {
        #'crf': 20,
        'preset': 'ultrafast'
    }

    # Get the video stream
    input_video_stream = next(s for s in in_container.streams if s.type == 'video')

    # Get the codec context
    input_codec_context = input_video_stream.codec_context

    # Copy some input properties
    codec_context.pix_fmt = input_codec_context.pix_fmt
    codec_context.height = input_codec_context.height
    codec_context.width = input_codec_context.width

    # # # Decode the packet into a frame
    # # frame = packet.decode_one()

    # for packet in in_container.demux():
    #     # Do something with `packet`, often:
    #     for frame in packet.decode():
    #         # Do something with `frame`.
    #         # Encode the frame using libx264
    #         for packet in codec_context.encode(frame):
    #             out_stream.write(packet)
    #             out_container.mux(packet)

    for frame in in_container.decode(video=0):
        # Encode the frame using libx264
        for packet in codec_context.encode(frame):
            #out_stream.write(packet)
            out_container.mux(packet)

    out_stream.close()
    # Close the output container
    out_container.close()

    # Get byte array from output
    bw_byte_array = output.getvalue()
    return bw_byte_array


def convert_to_bw(byte_array: bytes) -> bytes:
    """Convert video to black & white."""
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
