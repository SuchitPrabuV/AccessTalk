import cv2
import mediapipe as mp

# Load MediaPipe hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Create one reusable hand detector
_hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)


def get_landmarks(frame):
    """
    Input:
        frame -> OpenCV frame in BGR format

    Output:
        Returns a list of 42 float values:
        [x0, y0, x1, y1, ... x20, y20]

        Returns None if no hand is detected.
    """

    # Convert BGR frame to RGB because MediaPipe requires RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    result = _hands.process(rgb)

    # If no hand is detected, return None
    if not result.multi_hand_landmarks:
        return None

    # Get the first detected hand
    hand_landmarks = result.multi_hand_landmarks[0]

    x_vals = [p.x for p in hand_landmarks.landmark]
    y_vals = [p.y for p in hand_landmarks.landmark]

    min_x = min(x_vals)
    min_y = min(y_vals)

    landmark_list = []

    for point in hand_landmarks.landmark:
        landmark_list.extend([point.x - min_x, point.y - min_y])

    # Draw the hand skeleton on the frame
    mp_draw.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS
    )

    return landmark_list


# Standalone test mode
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    print("Camera started. Show your hand. Press Q to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        landmarks = get_landmarks(frame)

        if landmarks is not None:
            cv2.putText(
                frame,
                f"Landmarks detected: {len(landmarks)} values",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        else:
            cv2.putText(
                frame,
                "No hand detected",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("Camera Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()