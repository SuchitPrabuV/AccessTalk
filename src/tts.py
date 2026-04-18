import cv2
import pyttsx3
import threading
import time

# Create the text-to-speech engine once
_engine = pyttsx3.init()

# Configure speaking speed and volume
_engine.setProperty("rate", 140)
_engine.setProperty("volume", 1.0)

# Prevent the same word from repeating every frame
_last_spoken = ""
_last_time = 0
COOLDOWN = 2.5


def _speak_async(word):
    """
    Speak the word in a background thread so the camera does not freeze.
    """
    _engine.say(word)
    _engine.runAndWait()


def speak_word(word, frame, confidence):
    """
    Draw prediction on the frame and speak it if confidence is high enough.

    Args:
        word: predicted label string
        frame: OpenCV frame
        confidence: prediction confidence from 0.0 to 1.0
    """

    global _last_spoken, _last_time

    current_time = time.time()

    # High confidence: show prediction
    if confidence >= 0.40:
        cv2.rectangle(frame, (10, 10), (420, 100), (0, 180, 0), -1)

        cv2.putText(
            frame,
            word,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            3
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence * 100:.0f}%",
            (20, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2
        )

    # Low confidence: show waiting state
    else:
        cv2.rectangle(frame, (10, 10), (320, 70), (80, 80, 80), -1)

        cv2.putText(
            frame,
            "Detecting...",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (220, 220, 220),
            2
        )

    # Speak only if:
    # - confidence is high
    # - word is new OR cooldown time passed
    if (
        confidence >= 0.40 and
        (word != _last_spoken or current_time - _last_time > COOLDOWN)
    ):
        _last_spoken = word
        _last_time = current_time

        thread = threading.Thread(
            target=_speak_async,
            args=(word,),
            daemon=True
        )
        thread.start()

    return frame


# Standalone test
if __name__ == "__main__":
    import numpy as np

    print("Testing TTS module. You should hear: Hello")

    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    speak_word("Hello", test_frame, 0.95)

    cv2.imshow("TTS Test", test_frame)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()