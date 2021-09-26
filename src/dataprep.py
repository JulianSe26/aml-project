import ast
import json
import math
import concurrent
from datetime import datetime

import pydicom
import concurrent.futures
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from pydicom.pixel_data_handlers import apply_modality_lut
from pydicom.pixel_data_handlers import apply_voi_lut

IMAGE_SUBDIR = "data"

class CxrCOCO:
    def __init__(self, base_path):
        self.data_path = Path(base_path).expanduser()
        self.image_csv = pd.read_csv(self.data_path / "train_image_level.csv")

        self.basic_coco = {
            'images': [],
            'annotations': [],
            'categories': [{
                'id': 1,
                'name': 'opacity'
            }]
        }

        print("Processing files and extract COCO format:\n")
        for index, image_row in tqdm(self.image_csv.iterrows(), total=len(self.image_csv)):
            image_id = index
            self.basic_coco['images'].append(self.construct_image_node(self.data_path, image_row['StudyInstanceUID'], image_id, image_row['id']))
            self.basic_coco['annotations'].extend(self.construct_ann_nodes(image_id, image_row['boxes']))

    def to_json_file(self, name="labels.json"):
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
                        'category_id': 1,
                        'iscrowd': 0,
                        'bbox': [max(box['x'], 0), max(box['y'], 0), box['width'], box['height']],
                        'area': box['width'] * box['height']
                    })

        return new_anns

    def construct_image_node(self, data_path: Path, study_id: str, image_id: int, instance_id: str):
        image_path = self.get_image_path(data_path, study_id, instance_id)
        pil_image = Image.open(image_path)
        width, height = pil_image.size
        return {
                'id': image_id,
                'file_name': str(image_path.relative_to(data_path)),
                'height': height,
                'width': width,
                'date_captured': datetime.today().strftime('%Y-%m-%d'),  # original date was anonymised
                'instance_id': instance_id
                }

    def get_image_path(self, data_path, study_id, instance_id: str) -> Path:
        study_path = data_path / IMAGE_SUBDIR / "train" / study_id
        image_paths = []
        for series_path in study_path.iterdir():
            if series_path.is_dir():
                [image_paths.append(image_path) for image_path in series_path.iterdir()]

        for single_path in image_paths:
            # image_id contains _image as suffix - remove for matching
            if instance_id[:-6] in single_path.name:
                return single_path

        raise Exception("path not found")


def to_pillow(dcm: pydicom.Dataset) -> Image:
    img_array = apply_modality_lut(dcm.pixel_array, dcm)
    img_array = apply_voi_lut(img_array, dcm)
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        img_array = np.amax(img_array) - img_array

    # every radiologist will kill us for that one:
    # might scale based on target, e.g. tissue, bone etc. results in better predictions
    img_array = img_array - np.min(img_array)
    img_array = img_array / np.max(img_array)

    img_array = (img_array * 255).astype(np.uint8)

    return Image.fromarray(img_array)


def split_list(lst, parts):
    n = math.ceil(len(lst) / parts)
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def convert_all_to_png(search_dir: Path, thread_count=32):
    paths = list(search_dir.glob('**/*.dcm'))
    if len(paths) < 1:
        raise Exception(f"No dicom files found, in specified search path: {search_dir}")
    executor_pool = concurrent.futures.ThreadPoolExecutor(thread_count)
    pending_futures = []
    chunks = list(split_list(paths, thread_count))

    for tid, next_chunk in enumerate(chunks):
        pending_futures.append(executor_pool.submit(convert_chunk, next_chunk, search_dir))

    concurrent.futures.wait(pending_futures)


def convert_chunk(chunk: list[Path], search_dir: Path):
    for single_path in chunk:
        dcm = pydicom.dcmread(single_path)
        pillow_img = to_pillow(dcm)
        new_path = search_dir / IMAGE_SUBDIR / single_path.with_suffix(".png").relative_to(search_dir)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        pillow_img.save(str(new_path))


def append_labels_to_coco(ann_file_path: Path, study_file_path: Path, image_file_path: Path):
    with open(ann_file_path, 'r') as infile:
        coco = json.load(infile)
    study_level = pd.read_csv(study_file_path)
    image_level = pd.read_csv(image_file_path)

    if 'images' in coco:
        for img in coco['images']:
            study_id = image_level[image_level["id"] == img["instance_id"]]["StudyInstanceUID"]
            img['labels'] = np.squeeze((study_level[study_level["id"] == study_id.values[0] + "_study"].values[:, [1, 2, 3, 4]])).tolist()

    with open(ann_file_path, 'w') as outfile:
        json.dump(coco, outfile)
