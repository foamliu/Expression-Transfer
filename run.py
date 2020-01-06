import os
import pickle
import sys

import numpy as np
from skimage import io

sys.path.append('..')
from face3d import mesh
from face3d.morphable_model import MorphabelModel

s = 8e-04
# angles = [10, 30, 20]
angles = [0, 0, 0]
t = [0, 0, 0]
h = w = 256
c = 3
save_folder = 'results'


def transfer(alpha_exp):
    ep[:10] = alpha_exp
    vertices = bfm.generate_vertices(sp, ep)

    transformed_vertices = bfm.transform(vertices, s, angles, t)
    projected_vertices = transformed_vertices.copy()
    image_vertices = mesh.transform.to_image(projected_vertices, h, w)
    image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)
    return image


if __name__ == '__main__':
    bfm = MorphabelModel('data/BFM.mat')
    print('init bfm model success')

    sp = bfm.get_shape_para('zero')
    ep = bfm.get_exp_para('zero')
    tp = bfm.get_tex_para('zero')
    colors = bfm.generate_colors(tp)
    colors = np.minimum(np.maximum(colors, 0), 1)

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    with open('alpha_exp.pkl', 'rb') as fp:
        data = pickle.load(fp)

    for i, alpha_exp in enumerate(data):
        image = transfer(alpha_exp)
        io.imsave('{}/{}.jpg'.format(save_folder, i), image)
