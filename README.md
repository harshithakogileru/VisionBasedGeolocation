# Vision-Based Geolocation using MixVPR

A vision-based geolocation system that estimates the geographical location of a query image using Visual Place Recognition (VPR).

The project uses a pretrained MixVPR model for image retrieval, CLAHE preprocessing for the final pipeline, FAISS for similarity search, and GPS information from database images to estimate the location of a query image.

## Overview

The system follows the pipeline:

Query Image
    ↓
CLAHE Preprocessing
    ↓
Pretrained MixVPR
    ↓
Global Image Descriptor
    ↓
FAISS Similarity Search
    ↓
Top-K Database Matches
    ↓
GPS Coordinates
    ↓
Weighted GPS Estimation
    ↓
Predicted Latitude & Longitude

## Features

- Visual Place Recognition using pretrained MixVPR
- CLAHE-based image preprocessing
- ResNet50 backbone
- 4096-dimensional global image descriptors
- FAISS-based image retrieval
- Top-K similarity search
- GPS-based location estimation
- Recall@1, Recall@2 and Recall@3 evaluation
- FastAPI backend for prediction
- Flutter mobile application for image upload and location display

## Model

The project uses a **pretrained MixVPR model** obtained from an existing paper implementation.

The model is used for inference and was not trained from scratch as part of this project.

### Model Configuration

- Backbone: ResNet50
- Aggregator: MixVPR
- Input image size: 320 × 320
- Output descriptor dimension: 4096
- Mix depth: 4
- Output rows: 4

The pretrained checkpoint is stored at:

```text
backend/checkpoints/mixvpr.ckpt
