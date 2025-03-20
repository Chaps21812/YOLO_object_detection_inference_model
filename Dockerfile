FROM python:3.12.2
# FROM nvidia/cuda
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y libgl1
RUN pip install -r requirements.txt
RUN pip install opencv-python-headless
EXPOSE 30501

# Set the entry point
ENTRYPOINT ["python", "/app/YOLO_model/main.py"]