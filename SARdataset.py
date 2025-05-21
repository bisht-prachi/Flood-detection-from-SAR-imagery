import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomGamma(p=0.2),
    # A.Affine(scale=(0.8, 1.2), translate_percent=(0, 0.1), rotate=(-45, 45), shear=(-10, 10), p=0.5),
    A.GaussianBlur(blur_limit=3, p=0.2),
    # A.OpticalDistortion(p=0.3),
    # A.GridDistortion(p=.1),
    # A.HueSaturationValue(p=0.3),
    # A.CLAHE(p=0.3),
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

])


test_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

])

class SARdataset(Dataset):
  def __init__(self, df, split, transform=None):
    self.df = df
    self.split= split
    self.transform = transform

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    df_row = self.df.iloc[index]

    vv_image = cv2.imread(df_row['vv_image_path'], 0) / 255.0
    vh_image = cv2.imread(df_row['vh_image_path'], 0) / 255.0

    rgb_image = to_rgb(vv_image, vh_image)

    image_set = {}
    if self.split == 'test':
      image_set['image'] = rgb_image.transpose((2,0,1)).astype('float32')
      #only image
    else:
      flood_mask = cv2.imread(df_row['flood_label_path'], 0) / 255.0

      if self.transform:
        augmented = self.transform(image=rgb_image, mask=flood_mask)
        rgb_image = augmented['image']
        flood_mask = augmented['mask']

      image_set['image'] = rgb_image.transpose((2,0,1)).astype('float32')
      image_set['mask'] = flood_mask.astype('int64')

    return image_set