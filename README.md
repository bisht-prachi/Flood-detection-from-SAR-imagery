#  Flood Detection from SAR Imagery

This repository contains my submission for the **NASA Earth Science and Technology Competition Initiative (ETCI) Flood Detection Challenge**, focused on detecting flood events using **Sentinel-1 Synthetic Aperture Radar (SAR)** imagery. The challenge is organized by the **NASA Interagency Implementation and Advanced Concepts Team (IMPACT)** and the **IEEE GRSS Earth Science Informatics Technical Committee**.

##  Overview

The goal is to build a model that accurately detects flooded areas using SAR data, which is resilient to cloud cover and lighting conditions. This project builds upon the beginner walkthrough and starter notebook provided by [FloodBase](https://medium.com/cloud-to-street/jumpstart-your-machine-learning-satellite-competition-submission-2443b40d0a5a), with additional experimentation and enhancements.



##  Methodology

- Utilized **Sentinel-1 SAR (VV/VH)** time-series imagery.
- Preprocessed data to reduce speckle noise and normalize intensities.
- Trained a **U-Net-based CNN** for pixel-wise flood segmentation.
- Employed **data augmentation** and early stopping for robust performance.
- Evaluated using **Intersection over Union (IoU)** and **F1-score**.


 Results
Model evaluation on the public test set:

IoU: 0.54


Acknowledgements
[NASA IMPACT](https://nasa-impact.github.io/etci2021/)

IEEE GRSS ESI TC

[FloodBase beginner guide](https://medium.com/cloud-to-street/jumpstart-your-machine-learning-satellite-competition-submission-2443b40d0a5a)
