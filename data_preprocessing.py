import os
import numpy as np
from PIL import Image, ImageEnhance


def expand_data_volume(data_dir):
    # 好的弯管图像路径
    good_img_dir = os.path.join(data_dir, 'good')
    # 有瑕疵的弯管图像路径
    bad_img_dir = os.path.join(data_dir, 'bad')
    # 好的弯管图像输出路径
    good_img_dst_dir = os.path.join(data_dir, 'train', 'good')
    # 有瑕疵的弯管图像输出路径
    bad_img_dst_dir = os.path.join(data_dir, 'train', 'bad')
    # 创建输出文件夹
    os.makedirs(good_img_dst_dir)
    os.makedirs(bad_img_dst_dir)
    # 处理图像
    handle_img(good_img_dir, good_img_dst_dir)
    handle_img(bad_img_dir, bad_img_dst_dir)


def handle_img(img_dir, dst_dir):
    image_list = os.listdir(img_dir)
    for img in image_list:
        file = os.path.join(img_dir, img)
        try:
            image = Image.open(file)
            # resize到YOLO处理的416
            img_resize = image.resize((416, 416))
            img_resize.save(os.path.join(dst_dir, 'resize_' + img))
            # 将resize后的图像翻转90度做扩展
            img_rotate = img_resize.rotate(90)
            img_rotate.save(os.path.join(dst_dir, 'rotate_90_' + img))
            # 将resize后的图像翻转180度做扩展
            img_rotate = img_resize.rotate(180)
            img_rotate.save(os.path.join(dst_dir, 'rotate_180_' + img))
            # 亮度调整
            img_enhance = ImageEnhance.Brightness(img_resize).enhance(np.random.uniform(0.5, 1.5))
            img_enhance.save(os.path.join(dst_dir, 'enhance_' + img))
        except IOError as e:
            print('error :', e)


if __name__ == '__main__':
    print('data_preprocessing begin...')
    # 图片的上级目录，下级包括good和bad
    data_dir = '/Volumes/QING/TONGJI/智能化方法'
    expand_data_volume(data_dir)
    print('data_preprocessing end')


    # good_img_dst_dir = os.path.join(data_dir, 'train', 'good')
    # os.makedirs(good_img_dst_dir)
    # image = Image.open(os.path.join(data_dir, '37.jpg'))
    # print(image.size)
    # img_resize = image.resize((416, 416))
    # print(img_resize.size)
    # img_resize.save(os.path.join(good_img_dst_dir, 'new.jpg'))
    # img_rotate = img_resize.rotate(90)
    # img_rotate.save(os.path.join(good_img_dst_dir, 'rotate.jpg'))
    # img_enhance = ImageEnhance.Brightness(img_resize).enhance(np.random.uniform(0.5, 1.5))
    # img_enhance.save(os.path.join(good_img_dst_dir, 'enhance.jpg'))
