import os
import json

### GESTURES JSON UTILS ###

def get_images_folder(images):
    if len(images) == 0:
        raise ValueError("No images provided")
    if not os.path.exists(images[0]):
        raise FileNotFoundError(f"Image file not found: {images[0]}")
    images_folder = os.path.dirname(os.path.abspath(images[0]))
    return images_folder

def get_gestures_dict(images_folder):
    list_file = list(filter(lambda f: f.endswith("json"), os.listdir(images_folder)))[0]
    if not list_file:
        print(f"No JSON file found in {images_folder}")
        return None
    with open(images_folder + "/" + list_file, "r") as f:
        gestures_dict = json.load(f)
    return gestures_dict

def get_index_from_label(gestures_dict, images:list, label:int):
    """
    Get the index from the label.
    """
    if gestures_dict is None:
        raise ValueError("No gestures dictionary provided")
    
    # Get images path and index
    images_name = gestures_dict[str(label)]
    images_folder = get_images_folder(images)
    image_path = images_folder + images_name + ".png"
    if image_path in images:
        img_index = images.index(image_path)
    else:
        print(f"Image not found: {image_path}")
        return None
    return img_index

def get_label_from_index(gestures_dict, images:list, index:int):
    """
    Get the label from the index.
    """
    if gestures_dict is None:
        raise ValueError("No gestures dictionary provided")
    
    # Get images path and index
    image_path = images[index]
    images_name = os.path.splitext(os.path.basename(image_path))[0]
    label = None
    for key, value in gestures_dict.items():
        if value == images_name:
            label = int(key)
            break
    return label