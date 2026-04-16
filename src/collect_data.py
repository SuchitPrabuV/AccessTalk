import os
import cv2

DATASET_DIR = 'dataset'
IMAGES_PER_CLASS = 100

GESTURES = [
    'Hello', 'ThankYou', 'Yes', 'No', 'Help',
    'I', 'You', 'What', 'Where', 'Water',
    'Name', 'Please', 'Sorry', 'Good', 'Stop'
]

os.makedirs(DATASET_DIR, exist_ok=True)

for gesture in GESTURES:
    os.makedirs(os.path.join(DATASET_DIR, gesture), exist_ok=True)

cap = cv2.VideoCapture(0)

for gesture in GESTURES:
    print(f'\nCollecting images for: {gesture}')
    print('Press Q when you are ready')

    while True:
        ret, frame = cap.read()

        cv2.putText(
            frame,
            f'Ready for {gesture} - Press Q',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.imshow('Collect Data', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    count = 0

    while count < IMAGES_PER_CLASS:
        ret, frame = cap.read()

        filename = os.path.join(DATASET_DIR, gesture, f'{count}.jpg')
        cv2.imwrite(filename, frame)

        cv2.putText(
            frame,
            f'{gesture}: {count}/{IMAGES_PER_CLASS}',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 100, 0),
            2
        )

        cv2.imshow('Collect Data', frame)
        cv2.waitKey(1)

        count += 1

    print(f'Finished {gesture}')
    print('Take your time to prepare the next gesture.')
    print('Press N when you are ready to continue.')

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            break

cap.release()
cv2.destroyAllWindows()

print('All gesture images collected.')