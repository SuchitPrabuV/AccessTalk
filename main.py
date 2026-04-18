import cv2
import numpy as np
import joblib

from src.camera import get_landmarks
from src.tts import speak_word

MODEL_PATH = "model/model.pkl"

# Load trained Random Forest model
print("Loading model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")

# Start webcam
cap = cv2.VideoCapture(0)

# Optional: set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("AccessTalk is running. Show an ISL gesture. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read from webcam.")
        break

    # Flip frame so movement feels natural
    # frame = cv2.flip(frame, 1)

    # Get 42 landmark values from Siddhesh's module
    landmarks = get_landmarks(frame)

    if landmarks is not None:
        # Convert into model input shape
        X = np.array([landmarks])

        # Predict gesture label
        predicted_word = model.predict(X)[0]

        # Get confidence score
        confidence = model.predict_proba(X).max()
        
        print(f"Prediction: {predicted_word}, Confidence: {confidence:.2f}")
        # Send prediction to Adhil's TTS + overlay module
        frame = speak_word(predicted_word, frame, confidence)

    else:
        cv2.putText(
            frame,
            "Show your hand to the camera",
            (70, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (180, 180, 180),
            2
        )

    cv2.imshow("AccessTalk MVP", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()