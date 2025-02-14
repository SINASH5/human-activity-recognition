# Human Activity Recognition

This project uses a pre-trained 3D ResNet model to recognize human activities in videos.

## Files
- `recognise_human_activity.py`: Main script for activity recognition.
- `model/`: Contains the pre-trained model and class labels.
- `test/`: Contains test videos.

## Usage
1. Place your video in the `test/` folder.
2. Run the script:
   ```bash
   python recognise_human_activity.py \
     --model model/resnet-34_kinetics.onnx \
     --classes model/action_recognition_kinetics.txt \
     --input test/video1.mp4