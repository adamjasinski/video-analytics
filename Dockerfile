FROM ultralytics/ultralytics:8.1.14-cpu

RUN apt update && \
        # TODO - remove ffmpeg CLI dependency once the transcoding issue in src/video_handler is fixed
        apt install -y ffmpeg

COPY src /app/src
COPY pyproject.toml /app
COPY requirements* /app

WORKDIR /app
#RUN pip install -r requirements-cpu.in
RUN pip install -r requirements-cpu.txt

EXPOSE 8502:8502
CMD streamlit run --server.port 8502 src/app.py