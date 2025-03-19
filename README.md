# YOLO Containerized Model

This is a containerized model of YOLO built to run and train with a cpu. It has functionality to convert coco datasets to yolo datasets and output an evaluation as well. 

# Usage
To use each of the commands, an example is provided below for ease of use. The arguments are (mode, model size (n,s,m,l,x), epochs/model directory)

## Build Container 

docker build -t yolo_inference .

## Dataset Conversion

docker run --rm  -v /your/data/folder:/app/data yolo_inference convert n 0

## Training

docker run --rm -v /your/data/folder:/app/data yolo_inference train m 20

## Evaluation

docker run --rm -v /your/data/folder:/app/data yolo_inference eval n /app/data/yolo_annotations/models_and_results/model.pt