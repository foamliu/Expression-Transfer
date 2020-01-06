from PIL import Image


if __name__ == '__main__':
    for i in range(97):
        old_name = "images/frame_{:03d}_delay-0.1s.gif".format(i)
        print(old_name)
        new_name = 'images/{}.png'.format(i)
        img = Image.open(old_name)
        # print(img.shape)
        img.save(new_name)



