from ultralytics import YOLO
from ultralytics.data.converter import convert_coco
from YOLO_Preprocess import process_fits_and_save_png
import torch
import shutil
import os
import json
import yaml

class YOLO_Machina():
    def __init__(self, data_path:str, model_size:str="m"):
        if model_size == "n": self.model = YOLO("yolo11n.pt")
        if model_size == "s": self.model = YOLO("yolo11s.pt")
        if model_size == "m": self.model = YOLO("yolo11m.pt")
        if model_size == "l": self.model = YOLO("yolo11l.pt")
        if model_size == "x": self.model = YOLO("yolo11x.pt")
        self.device = next(self.model.model.parameters()).device

        #create train/test/eval paths for data
        self.data_path = data_path
        self.yolo_data_path = self.data_path+"/yolo_annotations"
        self.train_path = self.data_path+"/yolo_annotations/train"
        self.test_path = self.data_path+"/yolo_annotations/test"
        self.eval_path = self.data_path+"/yolo_annotations/eval"
        self.model_path = self.data_path+"/yolo_annotations/models_and_results"
        if not os.path.exists(self.yolo_data_path): os.mkdir(self.yolo_data_path)
        if not os.path.exists(self.train_path): os.mkdir(self.train_path)
        if not os.path.exists(self.test_path): os.mkdir(self.test_path)
        if not os.path.exists(self.eval_path): os.mkdir(self.eval_path)
        if not os.path.exists(self.model_path): os.mkdir(self.model_path)

        # Write data.yaml
        data = {"path": self.data_path,
            "train": self.train_path ,
            "val": self.train_path , 
            "test": self.train_path ,
            "nc": 1 ,
            "names": ["Satellite"]}
        with open(self.data_path +"/data.yaml", "w") as file:
            yaml.dump(data, file, default_flow_style=False)    

        print(f"Model is using: {self.device}")
        print("Cuda Available:", torch.cuda.is_available())

    def convert_coco_to_yolo(self):
        old_annotations = self.data_path+"/annotations"
        old_images = self.data_path+"/images"
        temp_path =self.data_path+"/temp"
        
        #convert to coco
        convert_coco(labels_dir=old_annotations, save_dir=temp_path)
        process_fits_and_save_png(old_images,self.train_path+"/images")

        # Move all files from incorrect annotations folder into labels folder
        source_dir = temp_path+"/labels/annotations"
        destination_dir = self.train_path+"/labels"
        if not os.path.exists(destination_dir): os.mkdir(destination_dir)
        for file_name in os.listdir(source_dir):
            source_file = os.path.join(source_dir, file_name)
            destination_file = os.path.join(destination_dir, file_name)
            if os.path.isfile(source_file):  # Ensure it's a file, not a directory
                shutil.move(source_file, destination_file)
        shutil.rmtree(temp_path)

    def train(self, epochs=1000, imgsz=1200, batch=0.7):
        self.model.train(data=self.data_path+"/data.yaml", epochs=epochs,imgsz=imgsz,batch=batch)
        self.model.save(self.model_path+"/model.pt")

    def evaluate(self, model_path):
        self.model = YOLO(model_path)
        self.device = next(self.model.model.parameters()).device
    
        path = self.eval_path+"/images"
        self.model= self.model.eval()
        temp_results = self.model.predict(path)
        results = {}
        for item in temp_results:
            filename = item.path.split("/")[-1]
            results[filename] = item.to_df().to_dict("list")
        with open(self.model_path+"/outputs.json", "w") as f:
            json.dump(results, f)
        return results

if __name__ == "__main__":
    model = YOLO_Machina("/app/data/fits_with_sp3_ra_dec_v3_COCO_dataset_mini")
    model.convert_coco_to_yolo()
    model.train( epochs = 1)
    results = model.evaluate("/app/data/model.pt")

    print("Done!")

    
