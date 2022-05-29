import cv2
import numpy as np
import argparse
import glob
import os
import sys
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Extract Frame from Video')

# Training settings
parser.add_argument('--input', type=str, default='./extract/input/',
                    help='path to Video')
parser.add_argument('--output', default='./extract/output',
                    help='path to Output')
parser.add_argument('--base', type=str, default='step',
                    help='captured frame base on step or num2cap')
parser.add_argument('--step', type=int, default=24,
                    help='step of each frame captured')
parser.add_argument('--num2cap', type=int, default=2,
                    help='num of each video to captured')
parser.add_argument('--mode', type=str, default='FACE',
                    help='EM, FACE, LARGE, SMALL')

args = parser.parse_args()

# Load face detector
face_detector_dir = './data/haarcascades_cuda/'
face_detector = cv2.CascadeClassifier(face_detector_dir + 'haarcascade_frontalface_alt2.xml')
eye_detector = cv2.CascadeClassifier(face_detector_dir + 'haarcascade_eye.xml')
smile_detector = cv2.CascadeClassifier(face_detector_dir + 'haarcascade_smile.xml')


def main():
    process_one_folder(args.input, args.output)


# 定义保存图片函数
# image:要保存的图片名字
# addr；图片地址与相片名字的前部分
# num: 相片，名字的后缀。int 类型
def save_image(image, dir, video_name, num, is_real):
    if is_real:
        address = dir + "/" + video_name + "_" + str(num) + '.jpg'
    else:
        address = dir + "/" + video_name + "_" + str(num) + '.png'

    h, w, l = image.shape
    w = (w if w & 1 == 0 else w - 1)
    h = (h if h & 1 == 0 else h - 1)
    if h > w:
        x1 = 0
        x2 = w
        y1 = int((h - w) / 2)
        y2 = int((h + w) / 2)
    elif w > h:
        x1 = int((w - h) / 2)
        x2 = int((w + h) / 2)
        y1 = 0
        y2 = h
    else:
        x1 = 0
        x2 = w
        y1 = 0
        y2 = h

    cropped = image[y1:y2, x1:x2]
    width = int(256)
    height = int(256)
    dim = (width, height)
    resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(address, resized)


def process_eye_smile(image, video_dir, video_name, num, is_real):
    if is_real:
        address_eye = video_dir + "/" + video_name + "_" + str(num) + '_eye.jpg'
        address_smile = video_dir + "/" + video_name + "_" + str(num) + '_smile.jpg'
    else:
        address_eye = video_dir + "/" + video_name + "_" + str(num) + '_eye.png'
        address_smile = video_dir + "/" + video_name + "_" + str(num) + '_smile.png'

    eyes = detectEyes(image)
    if len(eyes) > 0:
        save_eye_smile(image, eyes, address_eye, True, 0)

    smiles = detectSmiles(image)
    if len(smiles) > 0:
        save_eye_smile(image, smiles, address_smile, False, 0)

def process_smart_face(image, video_dir, video_name, num, is_real):
    if is_real:
        address_sface = '{}/{}_{}_sface.jpg'.format(video_dir, video_name, num)
    else:
        address_sface = '{}/{}_{}_face.png'.format(video_dir, video_name, num)

    faces = detectFaces(image)

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    for (x1, y1, x2, y2) in faces:
        result = []
        roi_gray = gray[y1:y2, x1:x2]
        eyes = eye_detector.detectMultiScale(roi_gray, 1.3, 2)
        for (ex, ey, ew, eh) in eyes:
            result.append((x1 + ex, y1 + ey, x1 + ex + ew, y1 + ey + eh))

        smiles = smile_detector.detectMultiScale(roi_gray, 4, 5)
        for (sx, sy, sw, sh) in smiles:
            result.append((x1 + sx, y1 + sy, x1 + sx + sw, y1 + sy + sh))

        if len(smiles) > 0:
            save_eye_smile(image, result, address_sface, True, 0)



def process_face(image, video_dir, video_name, num, is_real, bias):
    faces = detectFaces(image)
    if is_real:
        address_face = '{}/{}_{}_face.jpg'.format(video_dir, video_name, num)
    else:
        address_face = '{}/{}_{}_face.png'.format(video_dir, video_name, num)
    if len(faces) > 0:
        save_eye_smile(image, faces, address_face, False, bias)


def save_eye_smile(image, m_array, file_name, is_eye, bias):
    img_h, img_w, img_l = image.shape

    m_array_b = m_array + np.array([-int(bias), -int(bias), int(bias), int(bias)])

    x1, y1, i, j = np.min(m_array_b, axis=0)
    i, j, x2, y2 = np.max(m_array_b, axis=0)

    x1 = x1 if x1 >= 0 else 0
    y1 = y1 if y1 >= 0 else 0
    x2 = x2 if x2 <= img_w else img_w
    y2 = y2 if y2 <= img_h else img_h

    w = x2 - x1
    h = y2 - y1

    y2 = (y2 if h & 1 == 0 else y2 - 1)
    x2 = (x2 if w & 1 == 0 else x2 - 1)

    w = x2 - x1
    h = y2 - y1

    if h > w:
        cut_x1 = int((x1 + x2 - h) / 2)
        cut_x2 = int((x1 + x2 + h) / 2)
        cut_y1 = y1
        cut_y2 = y2
        if cut_x2 - cut_x1 > img_w:
            cut_x1 = 0
            cut_x2 = img_w
            cut_y1 = int((y1 + y2 - img_w) / 2)
            cut_y2 = int((y1 + y2 + img_w) / 2)
    elif w > h:
        cut_x1 = x1
        cut_x2 = x2
        cut_y1 = int((y1 + y2 - w) / 2)
        cut_y2 = int((y1 + y2 + w) / 2)
        if cut_y2 - cut_y1 > img_h:
            if is_eye:
                cut_x1 = x1
                cut_x2 = x1 + img_h
            else:
                cut_x1 = int((x1 + x2 - img_h) / 2)
                cut_x2 = int((x1 + x2 + img_h) / 2)
            cut_y1 = 0
            cut_y2 = img_h
    else:
        cut_x1 = x1
        cut_x2 = x2
        cut_y1 = y1
        cut_y2 = y2

    if cut_x1 < 0:
        cut_x2 = cut_x2 - cut_x1
        cut_x1 = 0
    if cut_y1 < 0:
        cut_y2 = cut_y2 - cut_y1
        cut_y1 = 0
    if cut_x2 > img_w:
        cut_x1 = cut_x1 - (cut_x2 - img_w)
        cut_x2 = img_w
    if cut_y2 > img_h:
        cut_y1 = cut_y1 - (cut_y2 - img_h)
        cut_y2 = img_h

    cropped = image[cut_y1:cut_y2, cut_x1:cut_x2]
    width = int(256)
    height = int(256)
    dim = (width, height)
    resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(file_name, resized)


def detectFaces(image):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image  # if语句：如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图

    faces = face_detector.detectMultiScale(gray, 1.4, 6)  # 1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
    result = []
    for (x, y, width, height) in faces:
        result.append((x, y, x + width, y + height))
    return result


def detectEyes(image):
    faces = detectFaces(image)

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    result = []
    for (x1, y1, x2, y2) in faces:
        roi_gray = gray[y1:y2, x1:x2]
        eyes = eye_detector.detectMultiScale(roi_gray, 1.4, 5)
        for (ex, ey, ew, eh) in eyes:
            result.append((x1 + ex, y1 + ey, x1 + ex + ew, y1 + ey + eh))
    return result


def detectSmiles(image):
    faces = detectFaces(image)

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    result = []
    for (x1, y1, x2, y2) in faces:
        roi_gray = gray[y1:y2, x1:x2]
        smiles = smile_detector.detectMultiScale(roi_gray, 2, 4)
        for (sx, sy, sw, sh) in smiles:
            result.append((x1 + sx, y1 + sy, x1 + sx + sw, y1 + sy + sh))
    return result


def process_one_file(fileName, output_dir, is_real):
    (path, v_name) = os.path.split(fileName)
    v_name = v_name.split(".")[0]
    # 读取视频文件
    videoCapture = cv2.VideoCapture(fileName)
    # 通过摄像头的方式
    # videoCapture=cv2.VideoCapture(1)

    frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    step_base_on_num2cap = frame_count // args.num2cap
    active_step = args.step if args.base == 'step' else step_base_on_num2cap
    active_step = active_step * 4 if is_real is True else active_step
    # read frame
    success, frame = videoCapture.read()
    i = 0
    j = 0
    while success:
        i = i + 1
        if i % active_step == 0:
            j = j + 1
            if args.mode == 'EM':
                process_eye_smile(frame, output_dir, v_name, j, is_real)
            elif args.mode == 'FACE':
                process_smart_face(frame, output_dir, v_name, j, is_real)
            elif args.mode == 'LARGE':
                process_face(frame, output_dir, v_name, j, is_real, 200)
            elif args.mode == 'SMALL':
                process_face(frame, output_dir, v_name, j, is_real, 10)

            # save_image(frame, output_dir, v_name, j, is_real)
            # print('save image:', i)
        success, frame = videoCapture.read()
    videoCapture.release()


def get_subdir_list(path):
    dirs = []
    dbtype_list = os.listdir(path)
    for item in dbtype_list:
        if os.path.isdir(os.path.join(path, item)):
            if item[0] == ".":
                continue
            dirs.append(item)
    return dirs


def process_one_folder(input_dir, output_dir, is_real=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    search_str = '{}/*.mp4'.format(input_dir)
    print('Extracting DIR: {}'.format(input_dir))
    pbar = tqdm(glob.glob(search_str))
    for filename in pbar:
        process_one_file(filename, output_dir, is_real)

    dir_list = get_subdir_list(input_dir)
    for folder in dir_list:
        if folder == "fake":
            is_real = False
        new_input = os.path.join(input_dir, folder)
        new_output = os.path.join(output_dir, folder)
        process_one_folder(new_input, new_output, is_real)


main()
