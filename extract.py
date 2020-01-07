import cv2 as cv
import dlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from config import device
from retinaface.detector import Detector
from utils.ddfa import ToTensorGjz, NormalizeGjz, _parse_param
from utils.inference import crop_img, parse_roi_box_from_bbox, parse_roi_box_from_landmark


def extract(img_ori):
    rects = face_detector(img_ori, 1)
    rect = rects[0]
    # dets, landms = detector.detect_faces(img_ori)
    # det = dets[0]
    # bbox = [det[0], det[1], det[2], det[3]]
    # print('bbox: ' + str(bbox))
    # roi_box = parse_roi_box_from_bbox(bbox)
    # print('roi_box: ' + str(roi_box))

    # - use landmark for cropping
    pts = face_regressor(img_ori, rect).parts()
    pts = np.array([[pt.x, pt.y] for pt in pts]).T
    roi_box = parse_roi_box_from_landmark(pts)

    img = crop_img(img_ori, roi_box)

    img = cv.resize(img, (120, 120), interpolation=cv.INTER_LINEAR)
    input = transform(img).unsqueeze(0)
    input = input.to(device)

    with torch.no_grad():
        param = model(input)
        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

    # print('param: ' + str(param))
    p, offset, alpha_shp, alpha_exp = _parse_param(param)
    # print('alpha_exp: ' + str(alpha_exp))
    return alpha_exp, p


if __name__ == '__main__':
    filename_scripted = '3ddfa_scripted.pt'
    model = torch.jit.load(filename_scripted)

    cudnn.benchmark = True
    model = model.to(device)
    model.eval()

    face_detector = dlib.get_frontal_face_detector()
    dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
    face_regressor = dlib.shape_predictor(dlib_landmark_model)

    # detector = Detector()

    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    alpha_exp_list = []
    pose_list = []

    video = 'data/kuazhangbiaoqing.mp4'
    cap = cv.VideoCapture(video)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        try:
            alpha_exp, p = extract(frame)
            alpha_exp_list.append(alpha_exp)
            pose_list.append(p)
        except IndexError as err:
            print(err)

    import pickle

    print(len(alpha_exp_list))

    data = dict()
    data['alpha_exp'] = alpha_exp_list
    data['pose'] = pose_list

    with open('data.pkl', 'wb') as fp:
        pickle.dump(data, fp)
