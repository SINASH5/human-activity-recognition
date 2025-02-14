
# Human Activity Recognition with OpenCV and Deep Learning

![Human Activity Recognition Demo](demo.gif)

Recognize human activities in videos using a pre-trained 3D ResNet model trained on the Kinetics-400 dataset.

## Features
- Recognizes **400+ activities** (e.g., yoga, cooking, dancing).
- Real-time inference on videos or webcam streams.
- Uses OpenCV and a spatiotemporal 3D ResNet model.

## Prerequisites
- Python 3.7+
- OpenCV 4.1.2+

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/human-activity-recognition.git
cd human-activity-recognition
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Model and Labels
Place the following files in the `model/` folder:
- `resnet-34_kinetics.onnx`: Pre-trained model.
- `action_recognition_kinetics.txt`: Class labels.

## Usage

### Run on a Video File
```bash
python recognise_human_activity.py \
  --model model/resnet-34_kinetics.onnx \
  --classes model/action_recognition_kinetics.txt \
  --input test/video1.mp4
```

### Run on Webcam
```bash
python recognise_human_activity.py \
  --model model/resnet-34_kinetics.onnx \
  --classes model/action_recognition_kinetics.txt
```
Press `q` to quit.

## Project Structure
```
human-activity-recognition/
├── model/                   # Pre-trained model and labels
│   ├── resnet-34_kinetics.onnx
│   └── action_recognition_kinetics.txt
├── test/                    # Test videos
│   └── video1.mp4
├── recognise_human_activity.py  # Main script
├── requirements.txt         # Dependencies
└── README.md
```

## Dependencies
- `opencv-python>=4.5.1`: Video processing and model inference.
- `imutils>=0.5.4`: Frame resizing utilities.
- `numpy>=1.19.3`: Array manipulations.

## Troubleshooting

### Video Not Loading?
Ensure the video file exists in `test/` and is in MP4 format.

Convert videos using FFmpeg:
```bash
ffmpeg -i input.avi -vcodec libx264 output.mp4
```

### OpenCV Errors?
Upgrade OpenCV:
```bash
pip install --upgrade opencv-python
```

### Model Not Found?
Ensure `resnet-34_kinetics.onnx` and `action_recognition_kinetics.txt` are in the `model/` folder.

## License
MIT License. See LICENSE.

## Acknowledgments
- **Model**: PyImageSearch (Hara et al.’s 3D ResNet).
- **Dataset**: Kinetics-400.
