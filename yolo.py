########## Instance Segmentation and mask generation task using YOLOv8n ##########

# importing libraries
import argparse
import cv2
import numpy as np
from ultralytics import YOLO


model_path = 'yolov8n-seg.pt' 
predefined_objects = ['chair', 'couch', 'bed', 'dining table'] 


def load_model(model_path):
    """
    Loads the YOLO model from the specified path.

    :param model_path: path to the model
    :return: loaded YOLO model
    """
    return YOLO(model_path)


def load_input_image(image_path):
    """
    Loads the input image from the specified path.

    :param image_path: path to the input image
    :return: tuple of the loaded input image and its height and width
    """
    input_image = cv2.imread(image_path)
    height, width, _ = input_image.shape # channels are not used
    return input_image, height, width


def filter_objects_dict(model, filtered_objects):
    """
    Filters out unwanted objects from the original dictionary. 

    :param model: loaded YOLO model
    :param filtered_objects: list of objects that should be kept
    :return: dictionary containing of the relevant key-value pairs of predefined objects
    """
    return {key: value for key, value in model.names.items() if value in filtered_objects}


def instance_segmentation(model, input_image, filtered_objects_dict, device='cpu'):
    """
    Performs instance segmentation using YOLO model.

    :param model: loaded YOLO model
    :param input_image: loaded input image
    :param filtered_objects_dict: dictionary containing of the relevant key-value pairs of predefined objects
    :device: specifies the model should run on CPU
    :return: saves the resulting JPG image 
    """
    classes = list(filtered_objects_dict.keys()) # obtains keys from dict and stores is in a list
    return model.predict(input_image, save=True, classes=classes, device=device) # saving the output image

def generate_masks(output_image, width, height):
    """
    Performs mask image generation using YOLO model.
    
    Saves the masks as PNG files.

    :param output_image: image returned by the instance_segmentation() function
    :param width: width of the input image
    :param height: height of the input image
    """
    for object in output_image:
        for i, mask in enumerate(object.masks.data):
            mask = mask.numpy() * 255 # converts the pixels to black and white
            mask = cv2.resize(mask, (width, height)) # ensures the size of the mask image will match the input
            mask_file = f'mask_{i}.png'
            cv2.imwrite(mask_file, mask)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform instance segmentation and mask generation tasks using YOLO model.")
    parser.add_argument("--image-path", required=True, help="Path to the input image")
    
    args = parser.parse_args()
    model = load_model(model_path)

    input_image, height, width = load_input_image(args.image_path)

    filtered_objects_dict = filter_objects_dict(model, predefined_objects)

    output_image = instance_segmentation(model, input_image, filtered_objects_dict)
    
    generate_masks(output_image, width, height)

print("GENERATION OF MASK IMAGES WAS COMPLETED")




