import os
import numpy as np
import pandas as pd
from glob import glob
from utils import get_filename

def construct_dataframe(data_dir, include_flood_label=True):
    vv_image_paths = sorted(glob(os.path.join(data_dir, '**/vv/*.png'), recursive=True))
    vv_image_names = [get_filename(pth) for pth in vv_image_paths]
    region_name_dates = ['_'.join(n.split('_')[:2]) for n in vv_image_names]

    vh_image_paths, flood_label_paths, water_body_label_paths, region_names = [], [], [], []

    for i in range(len(vv_image_paths)):
        vh_image_name = vv_image_names[i].replace('vv', 'vh')
        vh_image_path = os.path.join(data_dir, region_name_dates[i], 'tiles', 'vh', vh_image_name)
        vh_image_paths.append(vh_image_path)

        if include_flood_label:
            flood_image_name = vv_image_names[i].replace('_vv', '')
            flood_label_path = os.path.join(data_dir, region_name_dates[i], 'tiles', 'flood_label', flood_image_name)
        else:
            flood_label_path = np.nan
        flood_label_paths.append(flood_label_path)

        water_body_label_name = vv_image_names[i].replace('_vv', '')
        water_body_label_path = os.path.join(data_dir, region_name_dates[i], 'tiles', 'water_body_label', water_body_label_name)
        water_body_label_paths.append(water_body_label_path)

        region_name = region_name_dates[i].split('_')[0]
        region_names.append(region_name)

    df = pd.DataFrame({
        'vv_image_path': vv_image_paths,
        'vh_image_path': vh_image_paths,
        'flood_label_path': flood_label_paths,
        'water_body_label_path': water_body_label_paths,
        'region': region_names
    })

    return df.sort_values(by=['vv_image_path']).reset_index(drop=True)
