import os
import cv2
import mediapipe as mp
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATASET_DIR = 'dataset'
MODEL_PATH = 'model/model.pkl'

os.makedirs('model', exist_ok=True)

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3
)

data = []
labels = []

for gesture in os.listdir(DATASET_DIR):
    folder = os.path.join(DATASET_DIR, gesture)

    if not os.path.isdir(folder):
        continue

    for image_name in os.listdir(folder):
        image_path = os.path.join(folder, image_name)

        img = cv2.imread(image_path)

        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            hand_landmarks = result.multi_hand_landmarks[0]
            x_vals = [p.x for p in hand_landmarks.landmark]
            y_vals = [p.y for p in hand_landmarks.landmark]

            min_x = min(x_vals)
            min_y = min(y_vals)

            row = []
            for point in hand_landmarks.landmark:
                row.extend([point.x - min_x, point.y - min_y])

            data.append(row)
            labels.append(gesture)

print(f'Total samples found: {len(data)}')

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=True,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f'Accuracy: {accuracy * 100:.2f}%')

joblib.dump(model, MODEL_PATH)

print(f'Model saved at: {MODEL_PATH}')