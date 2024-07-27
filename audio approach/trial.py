import cv2
import dlib
import numpy as np
from vosk import Model, KaldiRecognizer
import pyaudio

# Speech Recognition Initialization
model = Model("./Model/model2")
recognizer = KaldiRecognizer(model, 16000)

mic = pyaudio.PyAudio()
stream = mic.open(rate=16000, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=2048)  # Reduced chunk size
stream.start_stream()

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("./face_weights.dat")

# read the image
cap = cv2.VideoCapture(0)

# Initial lip region for reference
prev_lip_region = None
recognized_words = []  # Store recognized words throughout the session

while True:
    _, frame = cap.read()
    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find faces
    faces = detector(gray)

    for face in faces:
        # Create landmark object
        landmarks = predictor(image=gray, box=face)

        # Extract lip region
        mouth_left = landmarks.part(48).x
        mouth_right = landmarks.part(54).x
        mouth_top = min(landmarks.part(50).y, landmarks.part(51).y)
        mouth_bottom = max(landmarks.part(58).y, landmarks.part(59).y)

        lip_region = gray[mouth_top:mouth_bottom, mouth_left:mouth_right]

        # Ensure lip region is not empty and resize if necessary
        if lip_region.shape[0] == 0 or lip_region.shape[1] == 0:
            continue

        # Resize lip region to a fixed size (for better optical flow estimation)
        lip_region = cv2.resize(lip_region, (100, 50))

        # Draw circles at lip landmarks
        for i in range(48, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Calculate optical flow in lip region
        if prev_lip_region is not None:
            # Convert lip regions to float32 for optical flow calculation
            prev_lip_region = prev_lip_region.astype(np.float32)
            lip_region = lip_region.astype(np.float32)

            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prev_lip_region, lip_region, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            motion = np.mean(magnitude)

            # Check if motion exceeds a certain threshold
            if motion > 2:  # Threshold for considering as talking
                cv2.putText(frame, "Talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Not talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Update previous lip region
        prev_lip_region = lip_region

        # Read audio data from the stream
        data = stream.read(2048)
        if len(data) == 0:
            print("MIC NOT WORKING")
        if recognizer.AcceptWaveform(data):
            recognized_text = recognizer.Result()[14:-3]
            print(f"Speech Recognition: {recognized_text}")
            recognized_words.append(recognized_text)  # Store recognized words

    cv2.imshow(winname="Mouth", mat=frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

stream.stop_stream()
stream.close()
mic.terminate()
cap.release()

# Close all windows
cv2.destroyAllWindows()

# Summarize recognized words after closing the window
if recognized_words:
    paragraph = "During the session, the following words were spoken: "
    for word in recognized_words:
        paragraph += f"{word}, "
    paragraph = paragraph[:-2]  # Remove trailing comma
    print(paragraph)
