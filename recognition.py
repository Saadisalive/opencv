import cv2
import mediapipe as mp
from ultralytics import YOLO
import winsound
import pyttsx3
import webbrowser
import os

# ------------------ Setup -------------------
# Load YOLOv8 model
model = YOLO("yolov8n.pt")
COCO_CLASSES = model.names

# Mediapipe hands setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Voice engine
engine = pyttsx3.init()

# ------------------ Utils -------------------
def speak(text):
    engine.say(text)
    engine.runAndWait()

def play_alert():
    winsound.Beep(800, 120)

def open_youtube():
    webbrowser.open("https://youtube.com")

def open_google():
    webbrowser.open("https://google.com")

def lock_screen():
    os.system("rundll32.exe user32.dll,LockWorkStation")

def volume_up():
    os.system("nircmd.exe changesysvolume 2000")  # requires NirCmd tool

def volume_down():
    os.system("nircmd.exe changesysvolume -2000")

# ------------------ Hand Gesture Rules -------------------
def classify_gesture(hand_landmarks):
    tips = [hand_landmarks.landmark[i].y for i in [4, 8, 12, 16, 20]]
    knuckles = [hand_landmarks.landmark[i].y for i in [3, 6, 10, 14, 18]]

    fingers = [tips[i] < knuckles[i] for i in range(5)]

    # Gestures
    if fingers == [True, False, False, False, False]:
        return "Thumbs Up"
    elif fingers == [False, True, True, False, False]:
        return "Peace"
    elif fingers == [False, False, False, False, False]:
        return "Fist"
    elif fingers == [False, True, True, True, True]:
        return "Stop"
    elif fingers[0] and fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
        return "OK"
    elif fingers == [False, True, False, False, False]:
        return "Point"
    elif fingers == [True, False, False, False, True]:
        return "Call Me"
    elif fingers == [False, True, True, True, False]:
        return "Volume Up"
    elif fingers == [False, False, True, True, False]:
        return "Volume Down"
    return None

# ------------------ Gesture Actions -------------------
def handle_gesture(gesture):
    play_alert()
    speak(f"You showed {gesture}")
    print(f"🖐 Gesture detected: {gesture}")

    if gesture == "Thumbs Up":
        open_google()
    elif gesture == "Peace":
        open_youtube()
    elif gesture == "Stop":
        speak("Stopping program")
        exit()
    elif gesture == "Volume Up":
        volume_up()
    elif gesture == "Volume Down":
        volume_down()
    elif gesture == "Fist":
        lock_screen()

# ------------------ Face Detection -------------------
def detect_faces(frame):
    results = model(frame, verbose=False)
    face_detected = False

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = COCO_CLASSES[cls_id]
            if label == "person":
                face_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, face_detected

# ------------------ Main -------------------
def main():
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(max_num_hands=1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Face detection
        frame, face_detected = detect_faces(frame)
        if face_detected:
            speak("Hello, I see you!")

        # Hand detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = classify_gesture(hand_landmarks)
                if gesture:
                    handle_gesture(gesture)

        cv2.imshow("Smart Detection App", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
