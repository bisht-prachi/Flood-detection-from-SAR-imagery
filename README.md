#  Flood Detection from SAR Imagery

This repository contains my submission for the **NASA Earth Science and Technology Competition Initiative (ETCI) Flood Detection Challenge**, focused on detecting flood events using **Sentinel-1 Synthetic Aperture Radar (SAR)** imagery. The challenge is organized by the **NASA Interagency Implementation and Advanced Concepts Team (IMPACT)** and the **IEEE GRSS Earth Science Informatics Technical Committee**.

##  Overview

The goal is to build a model that accurately detects flooded areas using SAR data, which is resilient to cloud cover and lighting conditions. This project builds upon the beginner walkthrough and starter notebook provided by [FloodBase](https://medium.com/cloud-to-street/jumpstart-your-machine-learning-satellite-competition-submission-2443b40d0a5a), with additional experimentation and enhancements.



##  Methodology

- Utilized **Sentinel-1 SAR (VV/VH)** time-series imagery.
- Combined polarized images to isolate flood masks.
- Trained a **U-Net-based CNN** for pixel-wise flood segmentation.
- Employed **data augmentation**.
- Evaluated using **Intersection over Union (IoU)**.


 Results
Model evaluation on the public test set:

IoU: 0.54

Some test visualizations:
![image](https://github.com/user-attachments/assets/d24b31e6-26d1-429a-8041-e70ac1bae483)
![image](https://github.com/user-attachments/assets/214ca9ca-727c-4993-b726-739de667b270)
![image](https://github.com/user-attachments/assets/b4ea4e91-4079-4bfb-88bd-c2f3d4b0e2aa)
![image](https://github.com/user-attachments/assets/b550ee06-ccdc-48d6-93cc-0f2a3d6fef18)





Acknowledgements
[NASA IMPACT](https://nasa-impact.github.io/etci2021/)

IEEE GRSS ESI TC

[FloodBase beginner guide](https://medium.com/cloud-to-street/jumpstart-your-machine-learning-satellite-competition-submission-2443b40d0a5a)
[Data host courtesy FloodBase](https://drive.usercontent.google.com/download?id=14HqNW5uWLS92n7KrxKgDwUTsSEST6LCr&authuser=0)
