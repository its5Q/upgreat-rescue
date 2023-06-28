import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import time
import random
import numpy as np
from ultralytics import YOLO
from utils import tile_image, scale_image, calculate_overlap_percentage, merge_boxes
from copy import copy
from evaluate import evaluate_map
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def divide_chunks(l, n):
    for i in range(0, len(l), n): 
        yield l[i:i + n]

class Ensemble():
    def __init__(self, classifier_threshold = 0.35, detector_iou_threshold = 0.5, detector_conf = 0.5):
        self.classifier_threshold = classifier_threshold
        self.detector_iou_threshold = detector_iou_threshold
        self.detector_conf = detector_conf

        # Loading classifier
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4907, 0.4993, 0.4939], [0.1058, 0.1061, 0.1067])
        ])

        self.detector_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.classifier = models.mobilenet_v3_small()
        self.classifier = self.classifier.to(device)

        num_ftrs = self.classifier.classifier[3].in_features
        self.classifier.classifier[3] = nn.Linear(num_ftrs, 1)
        self.classifier.classifier.append(nn.Sigmoid())
        self.classifier = self.classifier.to(device)
        self.classifier.eval()

        self.classifier.load_state_dict(torch.load('./classifier_model/last.pt'))

        # Loading detector
        self.detector = YOLO('./runs/detect/train3/weights/best.pt')

    def detect(self, image: Image.Image):
        scaled_image = scale_image(image)
        tiles = tile_image(scaled_image)
        
        # Split tiles into batches and classify them
        person_tiles = []
        # for batch in divide_chunks(tiles, 64):
        tile_images = torch.stack([self.transform(tile[2]) for tile in tiles]).to(device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = self.classifier(tile_images)

        for tile, pred in zip(tiles, outputs):
            if pred[0] > self.classifier_threshold:
                person_tiles.append(tile)

        print(f'Person tiles: {len(person_tiles)}')

        # Visualize classifier results
        '''
        scaled_image2 = scaled_image.convert("RGBA")
        overlay = Image.new("RGBA", scaled_image2.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for tile in person_tiles:
            draw.rectangle(tile[0], fill=(255, 0, 0, 128))

        result = Image.alpha_composite(scaled_image2, overlay)
        result.save(f'./test_images/{random.randint(1,100000)}.png')
        '''
        
        # Prepare detector tiles from classifier tiles
        detector_tiles = []
        while person_tiles:
            tile = person_tiles.pop()
            tile_position = tile[0]
            tile_center_x = tile_position[0] + (tile_position[2] - tile_position[0]) / 2
            tile_center_y = tile_position[1] + (tile_position[3] - tile_position[1]) / 2

            left = tile_center_x * 2 - 320
            top = tile_center_y * 2 - 320
            right = tile_center_x * 2 + 320
            bottom = tile_center_y * 2 + 320

            if left < 0:
                left = 0
                right = left + 640
            elif right > image.width:
                left = image.width - 640
                right = image.width

            if top < 0:
                top = 0
                bottom = top + 640
            elif bottom > image.height:
                top = image.height - 640
                bottom = image.height

            detector_image = image.crop((left, top, right, bottom))
            #detector_image.show()
            detector_tiles.append(((left, top, right, bottom), (640, 640), detector_image))

            person_tiles = [tile2 for tile2 in person_tiles if calculate_overlap_percentage((left, top, right, bottom), (tile2[0][0] * 2, tile2[0][1] * 2, tile2[0][2] * 2, tile2[0][3] * 2)) <= 0.99]

        detected_objects = []

        if not detector_tiles:
            return detected_objects

        # for batch in divide_chunks(detector_tiles, 16):
        results = self.detector.predict([tile[2] for tile in detector_tiles], iou=self.detector_iou_threshold, conf=self.detector_conf, verbose=False, half=True)
        for res, tile in zip(results, detector_tiles):
            probs = [prob.item() for prob in res.boxes.conf]
            boxes = [box.cpu().tolist() for box in res.boxes.xyxy]
            for prob, box in zip(probs, boxes):
                label = 0
                detected_objects.append(
                    ((tile[0][0] + box[0], tile[0][1] + box[1], tile[0][0] + box[2], tile[0][1] + box[3]), prob, label)
                )

        print(f'Detector tiles: {len(detector_tiles)}')

        return merge_boxes(detected_objects)


if __name__ == "__main__":
    

    model = Ensemble(detector_conf=0.76, classifier_threshold=0.3)

    results = []

    # with open('eval.pkl', 'rb') as res_file:
    #    results = pickle.load(res_file)

    total_time = 0
    image_list = list(Path('./private/images').glob('*.JPG'))

    for image_path in image_list:
        image = Image.open(image_path)
        annotations = open(Path('./private/annotations') / (image_path.name[:-4] + '.txt')).read().splitlines()

        start_time = time.time()
        boxes = [list(map(float, annotation.strip().split()))[1:] for annotation in annotations if annotation]
        
        target = {'boxes': [], 'labels': []}
        for box in boxes:
            x, y, width, height = box
            left = (x - width / 2) * image.width
            top = (y - height / 2) * image.height
            right = left + (width * image.width)
            bottom = top + (height * image.height)

            target['boxes'].append((left, top, right, bottom))
            target['labels'].append(0)
        
        target['boxes'] = np.array(target['boxes'])
        target['labels'] = np.array(target['labels'])

        prediction = {'boxes': [], 'scores': [], 'labels': []}
        model_pred = model.detect(image)

        # print(model_pred)

        for pred in model_pred:
            prediction['boxes'].append(pred[0])
            prediction['scores'].append(pred[1])
            prediction['labels'].append(pred[2])

        prediction['boxes'] = np.array(prediction['boxes'])
        prediction['scores'] = np.array(prediction['scores'])
        prediction['labels'] = np.array(prediction['labels'])

        results.append((target, prediction))

        end_time = time.time()
        print(f'Time elapsed: {(end_time - start_time):.5f}')
        
        total_time += end_time - start_time


    print(f'Total time: {total_time:.4f}')
    print(f'Average per image: {(total_time / len(image_list) * 1000):.2f} ms')
    print('Saving prediction results')
    with open('eval.pkl', 'wb') as res_file:
        pickle.dump(results, res_file)


    print(evaluate_map(results, iou_threshold=0.5, score_threshold=0.76))
    

        

        




