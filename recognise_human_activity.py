import numpy as np
import argparse
import imutils
import sys
import cv2

# Parse command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained human activity recognition model")
ap.add_argument("-c", "--classes", required=True, help="path to class labels file")
ap.add_argument("-i", "--input", type=str, default="", help="path to input video file")
args = vars(ap.parse_args())

# Load class labels
CLASSES = open(args["classes"]).read().strip().split("\n")
SAMPLE_DURATION = 16  # Number of frames to sample
SAMPLE_SIZE = 112     # Spatial dimensions (112x112)

# Load the human activity recognition model
print("[INFO] loading human activity recognition model...")
net = cv2.dnn.readNet(args["model"])

# Open the video file or webcam
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)

# Loop over frames from the video stream
while True:
    # Initialize a list to store frames
    frames = []

    # Read SAMPLE_DURATION frames
    for _ in range(SAMPLE_DURATION):
        (grabbed, frame) = vs.read()
        if not grabbed:
            print("[INFO] no frame read from stream - exiting")
            sys.exit(0)

        # Resize the frame and add it to the frames list
        frame = imutils.resize(frame, width=400)
        frames.append(frame)

    # Prepare the blob for inference
    blob = cv2.dnn.blobFromImages(frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE),
                                 (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)

    # Pass the blob through the network
    net.setInput(blob)
    outputs = net.forward()

    # Get the predicted class label
    label = CLASSES[np.argmax(outputs)]

    # Display the predicted activity on each frame
    for frame in frames:
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        cv2.imshow("Human Activity Recognition", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
vs.release()
cv2.destroyAllWindows()