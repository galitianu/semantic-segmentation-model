FROM python:3.9-slim-buster

WORKDIR /python-docker

COPY requirements.txt requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 gcc python3-dev -y

RUN pip3 install -r requirements.txt

ARG GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

COPY . .

CMD [ "python3", "gradio_ui.py" ]