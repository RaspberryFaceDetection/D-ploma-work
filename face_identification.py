import cv2
import numpy as np
import glob
import os
import dlib
import time

from gpiozero import MotionSensor

from settings import settings
from sftp import sftp

pir = MotionSensor(settings.CONFIG['MOTION_SENSOR_PIN'])


def motion_detected() -> bool:
    """
    Detect motion based on pir sensor

    :return: True if motion detected
    """

    while True:
        pir.wait_for_motion()
        return True


def detect(video_capture, saved_face_descriptor, names, detect_faces, predictor, face_rec):
    """
    Face detection algorithm

    :param video_capture:
    :param saved_face_descriptor:
    :param names:
    :param detect_faces:
    :param predictor:
    :param face_rec:
    :return:
    """
    start_time = time.time()
    while (time.time() - start_time) <= settings.CONFIG['SECONDS_TO_CAPTURE_FRAMES']:
        # Grab a single frame from WebCam
        ret, frame = video_capture.read()

        # Find all the faces and face encodings in the frame
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
                match = list((np.linalg.norm(saved_face_descriptor - (np.array(face_descriptor)), axis=1)))
                val, idx = min((val, idx) for (idx, val) in enumerate(match))

                name = "Unknown"
                if val < settings.CONFIG['FACE_LANDMARK_THRESHOLD']:
                    name = names[idx]

                print(name)

                if name == "Unknown":
                    intruder_time = time.strftime('%Y.%m.%d %H:%M:%S')
                    file_name = f'Unknown_{intruder_time}.png'
                    cv2.imwrite(file_name, frame)
                    folder_name = time.strftime('%Y_%m_%d')
                    if not sftp.sftp_path_exists(f'/upload/{folder_name}'):
                        sftp.mkdir(f'/upload/{folder_name}')
                    sftp.put_file_on_sftp(file_name, f'/upload/{folder_name}/{file_name}')
                    print(f'Unknown person enter on {intruder_time}')
                else:
                    return


def main():
    """
    Start face detection

    :return:
    """
    detect_faces = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_rec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, settings.CONFIG['PICTURE_WIDTH_RESOLUTION'])
    video_capture.set(4, settings.CONFIG['PICTURE_HIGH_RESOLUTION'])

    saved_face_descriptor = []
    names = []

    for face in glob.glob(os.path.join(os.getcwd(), "face_encodings", "*.npy")):
        temp = np.load(face)
        saved_face_descriptor.append(temp)
        names.append(os.path.basename(face[:-4]))

    while True:
        if motion_detected():
            detect(video_capture, saved_face_descriptor, names, detect_faces, predictor, face_rec)

    video_capture.release()


if __name__ == '__main__':
    main()
