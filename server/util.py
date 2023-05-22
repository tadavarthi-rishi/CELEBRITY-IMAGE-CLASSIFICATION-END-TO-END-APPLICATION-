import joblib
import json
import numpy as np
import base64
import cv2
from PIL import Image
from io import BytesIO
from wavelet import w2d

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []
    for img in imgs:
        scalled_raw_img = np.array(img.resize((32, 32))).reshape(32 * 32 * 3, 1)
        img_har = w2d(img, 'db1', 5)
        img_har_resized = np.array(Image.fromarray(img_har).resize((32, 32))).reshape(32 * 32, 1)
        combined_img = np.vstack((scalled_raw_img, img_har_resized))

        len_image_array = 32 * 32 * 3 + 32 * 32

        final = combined_img.reshape(1, len_image_array).astype(float)
        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.around(__model.predict_proba(final) * 100, 2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./artifacts/saved_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")


def get_pil_image_from_base64_string(b64str):
    encoded_data = b64str.split(',')[1]
    decoded_data = base64.b64decode(encoded_data)
    image = Image.open(BytesIO(decoded_data))
    return image

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = Image.open(image_path)
    else:
        img = get_pil_image_from_base64_string(image_base64_data)

    gray = img.convert('L')
    faces = face_cascade.detectMultiScale(np.array(gray), 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray.crop((x, y, x + w, y + h))
        roi_color = img.crop((x, y, x + w, y + h))
        eyes = eye_cascade.detectMultiScale(np.array(roi_gray))
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    return cropped_faces

def get_b64_test_image_for_virat():
    with open("b64.txt") as f:
        return f.read()

if __name__ == '__main__':
    load_saved_artifacts()

    print(classify_image(get_b64_test_image_for_virat(), None))
    print(classify_image(None, "./test_images/federer1.jpeg"))
    print(classify_image(None, "./test_images/federer2.jpeg"))
    print(classify_image(None, "./test_images/virat1.jpeg"))
    print(classify_image(None, "./test_images/virat2.jpeg"))
    print(classify_image(None, "./test_images/virat3.jpeg"))
    print(classify_image(None, "./test_images/serena1.jpeg"))
    print(classify_image(None, "./test_images/Sharapova2.jpeg"))
    print(classify_image(None, "./test_images/sharapova1.jpeg"))

