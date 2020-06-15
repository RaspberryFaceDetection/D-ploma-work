import os

import cv2
import numpy as np
import dlib
import argparse

from sftp import sftp


def save_images(files, detect_faces, predictor, face_rec):
    """
    Save images

    :param files: files
    :param detect_faces: detect faces
    :param predictor: predictor
    :param face_rec: face recognition
    :return:
    """
    for file in files:
        frame = cv2.imread(file)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces.detectMultiScale(gray, 1.3, 5)
        face_locations = []
        person_name, file_extension = os.path.splitext(file)

        for (x, y, w, h) in faces:
            face_locations.append(dlib.rectangle(x, y, x + w, y + h))

        if 0 < len(face_locations) < 2:
            for face_location in face_locations:
                face_encoding = predictor(frame, face_location)
                face_descriptor = face_rec.compute_face_descriptor(frame, face_encoding, 1)
                np.save("face_encodings/" + person_name, np.array(face_descriptor))
        else:
            print("No face or more then one face detected. Please check the image file")


def main():
    """
    Save new face

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, help="SFTP folder with images", required=True)

    args = parser.parse_args()

    folder = args.folder
    images = []
    for image in sftp.list_dir(folder):
        sftp.copy_sftp_to_local(f'{folder}/{image}', image)
        images.append(image)

    detect_faces = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_rec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

    save_images(images, detect_faces, predictor, face_rec)


if __name__ == '__main__':
    main()
