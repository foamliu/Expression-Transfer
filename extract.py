import cv2 as cv
import dlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from tqdm import tqdm

from config import device
from utils.ddfa import ToTensorGjz, NormalizeGjz, _parse_param
from utils.inference import crop_img, parse_roi_box_from_landmark


def extract(filename):
    img_ori = cv.imread(filename)
    img_ori = cv.cvtColor(img_ori, cv.COLOR_BGR2RGB)
    rects = face_detector(img_ori, 1)
    rect = rects[0]
    # bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
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
    return alpha_exp


if __name__ == '__main__':
    filename_scripted = '3ddfa_scripted.pt'
    model = torch.jit.load(filename_scripted)

    cudnn.benchmark = True
    model = model.to(device)
    model.eval()

    face_detector = dlib.get_frontal_face_detector()
    dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
    face_regressor = dlib.shape_predictor(dlib_landmark_model)

    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    alpha_exp_list = []

    for i in tqdm(range(97)):
        filename = 'data/{}.png'.format(i)
        try:
            alpha_exp = extract(filename)
            alpha_exp_list.append(alpha_exp)
        except IndexError as err:
            print(err)

    import pickle

    with open('alpha_exp.pkl', 'wb') as fp:
        pickle.dump(alpha_exp_list, fp)
