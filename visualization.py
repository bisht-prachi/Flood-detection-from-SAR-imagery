import cv2
import pandas as pd
import matplotlib.pyplot as plt
from utils import to_rgb

def visualize(df_row):
    """
    Visualizes the RGB image along with water body and flood masks.
    """
    vv_image = cv2.imread(df_row['vv_image_path'], 0) / 255.0
    vh_image = cv2.imread(df_row['vh_image_path'], 0) / 255.0
    water_body_label = cv2.imread(df_row['water_body_label_path'], 0) / 255.0
    rgb_image = to_rgb(vv_image, vh_image)

    if pd.isna(df_row['flood_label_path']):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_image)
        plt.title("RGB Image")
        plt.subplot(1, 2, 2)
        plt.imshow(water_body_label)
        plt.title("Water Body")
    else:
        flood_label = cv2.imread(df_row['flood_label_path'], 0) / 255.0
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(rgb_image)
        plt.title("RGB Image")
        plt.subplot(1, 3, 2)
        plt.imshow(flood_label)
        plt.title("Flood Mask")
        plt.subplot(1, 3, 3)
        plt.imshow(water_body_label)
        plt.title("Water Body")
    plt.tight_layout()
    plt.show()



def visualize_result(df_row, prediction, figsize=[25, 15]):
    vv_image = cv2.imread(df_row['vv_image_path'], 0) / 255.0
    vh_image = cv2.imread(df_row['vh_image_path'], 0) / 255.0
    rgb_input = to_rgb(vv_image, vh_image)

    plt.figure(figsize=tuple(figsize))
    plt.subplot(1,2,1)
    plt.imshow(rgb_input)
    plt.title('RGB w/ result')
    plt.subplot(1,2,2)
    plt.imshow(prediction)
    plt.title('Result')