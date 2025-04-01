from ultralytics import YOLO
import cv2
import json
import numpy as np

category_map = {}
def main(annotations_path):
    # Measurements in meters
    # Pitch 120x90
    world_points = [
    {"Left Corner Flag": [0,0]},   # Left corner flag
    {"Top Left 18-Box": [24.85,0]},  # Top Left 18-Box
    {"Top Left 12-Box" : [35.85,0]},  # Top Left 12-Box
    {"Bottom Left 18-Box" : [24.85, 16.5]}, # Bottom Left 18-Box
    {"Bottom Left 12-Box" : [35.85, 5.5]}, # Bottom Right 12-Box
    {"Left Goal Post" : [41.34, 0]},  # Left Goal Post
    {"Right Goal Post" : [48.66, 0]},  # Right Goal Post
    {"Top Right 12-Box" : [54.15, 0]},   # Top Right 12-Box
    {"Top Right 18-Box" : [65.15, 0]},  # Top Right 18-Box
    {"Bottom Right 12-Box" : [54.15, 5.5]},  # Bottom Right 12-Box
    {"Bottom Right 18-Box" : [65.15, 16.5]},  # Bottom Right 18-Box
    {"Right Corner Flag" : [90, 0]}  # Right corner flag
    ]
    bbox_info = extract_coordinates(annotations_path=annotations_path)
    homography_matrix = calculate_homography_matrix(bbox_info=bbox_info, world_points=world_points)
    transformed_bbox_info = convert_pitch_coordinates(homography_matrix, bbox_info)
    print(transformed_bbox_info)


def convert_pitch_coordinates(homography_matrix, bbox_info):
    transformed_bbox_info = {}
    for feature in bbox_info:
        for label, coords in feature.items():
            transformed_bbox_info[label] = []
            x,y,w,h = coords[0], coords[1], coords[2], coords[3]
            bottom_center_x = x + w / 2
            bottom_center_y = y + h
            homogeneous_point = np.array([bottom_center_x, bottom_center_y, 1])
            transformed_point = np.dot(homography_matrix, homogeneous_point)
            world_x = transformed_point[0] / transformed_point[2]
            world_y = transformed_point[1] / transformed_point[2]
            transformed_bbox_info[label] = [world_x, world_y]
    return transformed_bbox_info
def predict_players_positioning(path_to_model, img_path):
    model = YOLO(path_to_model)
    # Assume we can pick up the ball
    results = model(img_path)
    
def calculate_homography_matrix(bbox_info, world_points):
    """
    Arguments: 
        bbox_info(list) - this is a list of dictionaries that maps labels:points [x,y,w,h] of pitch featres
        world_points(list) - list of dictionaries that maps labels:points[x,y,w,h] of standardizes pitch features
    Returns:
        matrix(np.array) - a homography matrix used to convert 3d pitch coordinates onto a 2d plane
        
    """
    image_points = []
    world_points_transformed = []
    world_points_dict = {list(item.keys())[0]: list(item.values())[0] for item in world_points}
    
    for item in bbox_info:
        feature_name = list(item.keys())[0]
        bbox = list(item.values())[0]
        mid_x = bbox[0] + bbox[2] / 2
        mid_y = bbox[1] + bbox[3] / 2
        image_points.append([mid_x, mid_y])
        world_points_transformed.append(world_points_dict[feature_name])

    image_points = np.array(image_points, dtype=np.float32)
    world_points_transformed = np.array(world_points_transformed, dtype=np.float32) 
    
    # Calculate homography matrix
    matrix, _ = cv2.findHomography(image_points, world_points_transformed)
    return matrix


def extract_coordinates(annotations_path):
    """
    Arguments: 
        annotations_path(str)- a path to an annotations json folder used to extract categories, and bbox coords
    Return:
        bbox_info(list) - a list of dictionaries that contains labels mapped to coordinates of key features
    """
    bbox_info = []
    data = {}
    with open(annotations_path, 'r') as file:
        data = json.load(file)
    # Build category map to map ids to labels
    # Only required once
    if not category_map:
        for category in data.get('categories', []):
            category_map[category['id']] = category['name']
    sorted_annotations = sorted(data.get("annotations", []), key=lambda x: x['category_id'])
    for annotation in sorted_annotations:
        category_id = annotation['category_id']
        bbox = annotation['bbox']
        bbox_info.append({category_map[category_id]: bbox})
    return bbox_info

if __name__ == "__main__":
    annotations_path = 'screenshots/annotations/instances_default.json'
    main(annotations_path=annotations_path)


