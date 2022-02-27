import os
import time


import torch
import numpy
import warnings
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt

from select_classes import *
from preprocess_video_frames import decompose_video_to_frames
from process_res_frames import format_result_frames


def process_frames(folder_name, classes_to_predict=['car']):
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.to('cpu')
    # set up the model
    model.classes = classes_to_predict

    # gather the images to be processed
    cnt = 0
    for img in sorted(os.listdir(folder_name)):
        if cnt % 10 == 0:
            print(' .frame {}'.format(cnt))

        img = Image.open(folder_name + '/' + img)
        img = img.resize([int(0.25 * s) for s in img.size])  # todo: explain the resize x speed
        
        result = model(img)
        
        assert len(result) == 1, 'not only one image at once processed'        
        result.render()

        plt.imshow(result.imgs[0])
        plt.savefig('results/image{}.jpg'.format(cnt))
        cnt += 1

        if cnt == 150:
            break
        
if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    video = 'data/giraffes.mp4'
    # classes_to_predict = ['car', 'motorcycle', 'carpet', 'person']
    classes_to_predict = ['giraffe']

    # create the results folder
    os.system('mkdir results')

    print('$> decomposing video to frames...')
    folder_name = decompose_video_to_frames(video)
    
    ctp = select_class(classes_to_predict)
    print('\n$> predict using yolo in the video the following classes: ', ctp)
    process_frames(folder_name, ctp)

    # print('$> recompose rendered frames into video footate')
    # format_result_frames()

    # delete the results folder ~ clean-up
    # os.system('rm -rf results')
