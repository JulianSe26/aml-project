import ast
import json

import pandas as pd
from pathlib import Path
import pydicom
from tqdm import tqdm


class CxrCOCO:
    def __init__(self, base_path):
        self.data_path = Path(base_path).expanduser()
        self.image_csv = pd.read_csv(self.data_path / "train_image_level.csv")

        self.basic_coco = {
            'images': [],
            'annotations': [],
            'categories': [{
                'id': 0,
                'name': 'opacity'
            }]
        }

        print("Processing dicom files and extract COCO format:\n")
        for index, image_row in tqdm(self.image_csv.iterrows(), total=len(self.image_csv)):
            image_id = index
            self.basic_coco['images'].append(self.construct_image_node(self.data_path, image_row['StudyInstanceUID'], image_id, image_row['id']))
            self.basic_coco['annotations'].extend(self.construct_ann_nodes(image_id, image_row['boxes']))

    def to_json_file(self, name):
        json_path = self.data_path / name
        print(f"Writing result to file: {json_path}")
        with open(json_path, 'w') as json_file:
            json.dump(self.basic_coco, json_file)

    def construct_ann_nodes(self, image_id: int, boxes_str) -> []:
        new_anns = []
        if not pd.isnull(boxes_str):
            boxes = ast.literal_eval(boxes_str)
            if isinstance(boxes, list):
                for index, box in enumerate(boxes):
                    new_anns.append({
                        'id': int(str(image_id) + str(index)),
                        'image_id': image_id,
                        'category_id': 0,
                        'iscrowd': 0,
                        'bbox': [box['x'], box['y'], box['width'], box['height']]
                    })

        return new_anns

    def construct_image_node(self, data_path: Path, study_id: str, image_id: int, instance_id: str):
        image_path = self.get_image_path(data_path, study_id, instance_id)
        dcm_image = pydicom.dcmread(image_path)
        return {
                'id': image_id,
                'file_name': str(image_path.relative_to(data_path)),
                'height': dcm_image.Rows,
                'width': dcm_image.Columns,
                'date_captured': dcm_image.StudyDate,
                'instance_id': instance_id
                }

    def get_image_path(self, data_path, study_id, instance_id: str) -> Path:
        study_path = data_path / "train" / study_id
        image_paths = []
        for series_path in study_path.iterdir():
            if series_path.is_dir():
                [image_paths.append(image_path) for image_path in series_path.iterdir()]

        for single_path in image_paths:
            # image_id contains _image as suffix - remove for matching
            if instance_id[:-6] in single_path.name:
                return single_path

        raise Exception("path not found")
