import os

import cv2
import numpy as np
import dlib
import argparse

from settings import settings
from sftp import sftp


def save_images(images, detect_faces, predictor, face_rec):
    """
    Save images

    :param images: images
    :param detect_faces: detect faces
    :param predictor: predictor
    :param face_rec: face recognition
    :return:
    """
    for image in images:
        frame = cv2.imread(image)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces.detectMultiScale(gray, 1.3, 5)
        face_locations = []
        person_name, file_extension = os.path.splitext(image)

        for (x, y, w, h) in faces:
            face_locations.append(dlib.rectangle(x, y, x + w, y + h))

        if 0 < len(face_locations) < 2:
            for face_location in face_locations:
                face_encoding = predictor(frame, face_location)
                face_descriptor = face_rec.compute_face_descriptor(
                    frame, face_encoding, 1
                )
                np.save(
                    f"{settings.CONFIG['FACE_ENCODINGS_DIR']}/{person_name}",
                    np.array(face_descriptor),
                )
        else:
            print("No face or more then one face detected. Please check the image file")


def main():
    """
    Save new face

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--folder", type=str, help="SFTP folder with images", required=True
    )

    args = parser.parse_args()

    folder = args.folder
    images = []
    for image in sftp.list_dir(folder):
        sftp.copy_sftp_to_local(f"{folder}/{image}", image)
        images.append(image)

    detect_faces = cv2.CascadeClassifier(settings.CONFIG["CASCADE_CLASSIFIER"])
    predictor = dlib.shape_predictor(settings.CONFIG["SHAPE_PREDICTOR"])
    face_rec = dlib.face_recognition_model_v1(settings.CONFIG["FACE_RECOGNITION_MODEL"])

    save_images(images, detect_faces, predictor, face_rec)


if __name__ == "__main__":
    main()
