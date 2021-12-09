import cv2
import numpy as np
import glob
import os
import dlib
import time

import requests
from gpiozero import MotionSensor

from settings import settings

pir = MotionSensor(settings.CONFIG['MOTION_SENSOR_PIN'])
detect_faces = cv2.CascadeClassifier(settings.CONFIG["CASCADE_CLASSIFIER"])
predictor = dlib.shape_predictor(settings.CONFIG["SHAPE_PREDICTOR"])
face_rec = dlib.face_recognition_model_v1(settings.CONFIG["FACE_RECOGNITION_MODEL"])

def send_to_telegram(a, b):
    pass


def motion_detected() -> bool:
    """
    Detect motion based on pir sensor

    :return: True if motion detected
    """

    while True:
        pir.wait_for_motion()
        return True


def detect_online(frame):
    url = f"http://{settings.CONFIG['INSTANCE_HOST']}:{settings.CONFIG['INSTANCE_PORT']}/get-recognition"
    file_name = f"verify_{time.strftime('%Y.%m.%d %H:%M:%S')}.jpg"
    cv2.imwrite(file_name, frame)
    files = {"file": (file_name, open(file_name, "rb"))}
    r = requests.post(url, files=files, timeout=5)
    return r.text


def detect_offline(frame, saved_face_descriptor, names):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = detect_faces.detectMultiScale(gray, 1.3, 5)

    face_locations = []
    for (x, y, w, h) in faces:
        face_locations.append(dlib.rectangle(x, y, x + w, y + h))

    # If there is at least 1 face then do face recognition
    if len(face_locations) > 0:
        for face_location in face_locations:
            face_encoding = predictor(frame, face_location)
            face_descriptor = face_rec.compute_face_descriptor(frame, face_encoding, 1)

            # See if the face is a match for the known face(s)
            match = list(
                (
                    np.linalg.norm(
                        saved_face_descriptor - (np.array(face_descriptor)), axis=1
                    )
                )
            )
            val, idx = min((val, idx) for (idx, val) in enumerate(match))

            name = settings.CONFIG["UNKNOWN_PERSON_NAME"]
            if val < settings.CONFIG["FACE_LANDMARK_THRESHOLD"]:
                name = names[idx]
            return name


def detect(video_capture, saved_face_descriptor, names):
    """
    Face detection algorithm

    :param video_capture:
    :param saved_face_descriptor:
    :param names:
    :return:
    """
    start_time = time.time()
    seconds_to_capture_frames = settings.CONFIG["SECONDS_TO_CAPTURE_FRAMES"]
    while (time.time() - start_time) <= seconds_to_capture_frames:
        # Grab a single frame from WebCam
        ret, frame = video_capture.read()
        send_message = False

        try:
            name = detect_online(frame)
            send_message = True
        except (requests.ConnectionError, requests.Timeout):
            name = detect_offline(frame, saved_face_descriptor, names)

        if send_message:
            send_to_telegram(name, frame)

        time.sleep(seconds_to_capture_frames)


def main():
    """
    Start face detection

    :return:
    """

    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, settings.CONFIG["PICTURE_WIDTH_RESOLUTION"])
    video_capture.set(4, settings.CONFIG["PICTURE_HIGH_RESOLUTION"])

    saved_face_descriptor = []
    names = []

    for face in glob.glob(
        os.path.join(os.getcwd(), settings.CONFIG["FACE_ENCODINGS_DIR"], "*.npy")
    ):
        temp = np.load(face)
        saved_face_descriptor.append(temp)
        names.append(os.path.basename(face[:-4]))

    while True:
        if motion_detected():
            detect(
                video_capture,
                saved_face_descriptor,
                names,
                detect_faces,
                predictor,
                face_rec,
            )

    video_capture.release()


if __name__ == "__main__":
    main()
