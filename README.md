Face Liveness Detection using MobileNetV3-Large
================================================

This project implements a deep-learning–based binary classification model to detect
real vs spoof faces using the Rose-Youtu Face Anti-Spoofing dataset. The model
uses MobileNetV3-Large and single-frame sampling from videos for efficient training.

------------------------------------------------------------
PROJECT OVERVIEW
------------------------------------------------------------
Face liveness detection determines whether a detected face is from a real person
or a spoof attack (photo, replay, mask, etc.). This project trains a lightweight
MobileNetV3-Large model to output:

1 -> Real (live)
0 -> Spoof (fake)

The model is trained using one random frame per video per epoch.

------------------------------------------------------------
DATASET
------------------------------------------------------------
Dataset: Rose-Youtu Face Anti-Spoofing Dataset
Total videos: 3497

Each filename encodes:
- Label (real/spoof)
- Attack type
- Person ID
- Video index

A custom Python script parses filenames and generates a metadata CSV:
rose_youtu_filelist.csv

CSV columns include:
- video_path
- filename
- person_id
- label
- attack_type
- split (train/test)

------------------------------------------------------------
FRAME SAMPLING STRATEGY
------------------------------------------------------------
Instead of reading entire videos, the dataset class:
- Opens a video with OpenCV
- Picks ONE random frame each time the video is sampled
- Converts it to RGB
- Resizes to 224×224
- Normalizes using ImageNet statistics

This improves efficiency and variation across epochs.

------------------------------------------------------------
MODEL INPUT SHAPE
------------------------------------------------------------
Batch of 32 images:
images = [32, 3, 224, 224]
labels = [32]

3 -> RGB channels
224x224 -> MobileNetV3 standard input size

------------------------------------------------------------
MODEL ARCHITECTURE
------------------------------------------------------------
We use MobileNetV3-Large pretrained on ImageNet:

- Initial convolution
- Multiple inverted residual bottleneck blocks
- Depthwise separable convolutions
- Squeeze-and-Excitation (SE) blocks
- Hard-Swish activation
- Global average pooling
- Final fully-connected layer replaced with: Linear(in_features, 1)

1 output + sigmoid = binary classification.

Loss:
BCEWithLogitsLoss

------------------------------------------------------------
TRAINING PIPELINE
------------------------------------------------------------
Training loop includes:
- Forward pass
- BCEWithLogitsLoss calculation
- Backpropagation
- Optimizer step
- Accuracy calculation
- Validation per epoch

Model is saved as:
checkpoints/mobilenetv3_liveness.pth

------------------------------------------------------------
PROJECT STRUCTURE
------------------------------------------------------------
FaceLivenessDetection/
│
├── build_rose_youtu_csv.py
├── train.ipynb
├── metadata/
│     └── rose_youtu_filelist.csv
├── videos_raw/
├── checkpoints/
│     └── mobilenetv3_liveness.pth
└── README.txt

------------------------------------------------------------
REQUIREMENTS
------------------------------------------------------------
python >= 3.8
torch >= 2.0
torchvision >= 0.15
opencv-python
pandas
Pillow

Install packages:
pip install torch torchvision opencv-python pandas pillow

------------------------------------------------------------
HOW TO RUN
------------------------------------------------------------
Step 1 — Generate metadata CSV:
python build_rose_youtu_csv.py --videos_dir ./videos_raw --output_csv ./metadata/rose_youtu_filelist.csv

Step 2 — Run training in train.ipynb

------------------------------------------------------------
CONTACT
------------------------------------------------------------
For issues, please open a GitHub issue.

