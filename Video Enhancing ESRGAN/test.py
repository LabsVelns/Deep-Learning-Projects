# -*- coding: utf-8 -*-

"""test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1azHZe_x19yb84JgWWZHNS0bxSWCqvnjg
"""

from collections import OrderedDict
import torch
import torch.nn as nn

import math

import os.path
import glob
import cv2
import numpy as np

import matplotlib.pyplot as plt
from Architecture import RRDB_Net


def process_video_with_esrgan(test_vid=None, model_path=None):
    # initialize pre-trained ESRGAN fine tuned for videos
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu',mode='CNA', res_scale=1, upsample_mode='upconv')
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)


    # switch to evaluate mode
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    print('Model path {:s}. \nProcessing Video...'.format(model_path))

    # iterate through all videos in test folder
    for path in glob.glob(test_vid):

        # start video capture
        cap = cv2.VideoCapture(test_vid)

        # Define the codec and create VideoWriter object
        base = os.path.splitext(os.path.basename(test_vid))[0]
        FPS = cap.get(cv2.CAP_PROP_FPS)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(r'C:\Users\harsh\Downloads\Project Run\{:s}_ESRGAN.avi'.format(base),
                              fourcc,
                              FPS,
                              (int(width * 4), int(height * 4)))

        # process video
        while(cap.isOpened()):

            # read a frame of the video
            ret, img = cap.read()
            if ret == True:

                # pre-process frame to expected model input format
                img = img * 1.0 / 255
                img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()   #[height,width,channel] = [channel,height,width]
                img_LR = img.unsqueeze(0) 
                img_LR = img_LR.to(device)

                # generate a super resolution frame
                output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
                output = (output * 255.0).round().astype(np.uint8)

                # write the super resolution frame frame
                out.write(output)
                cv2.imshow('Frame', output)

                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break


            else:
                print('Done {:s}'.format(base))
                break

        # Path for output video in Device
        pathop =r'C:\Users\harsh\Downloads\Project Run\{:s}_ESRGAN.avi'.format(base)

        # Release everything if job is 
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    return pathop
    

# 
# # Call the function with your test video folder and model path
# test_vid_folder = r'C:\Users\harsh\Downloads\*'
# model_path = r"C:\Users\harsh\Downloads\VID_tune2-2.pth"
# process_video_with_esrgan(test_vid_folder, model_path)














# import joblib
# # import io
# # def save_process_video_with_esrgan_model(process_video_with_esrgan):
# #   path = r'C:\Users\harsh\Downloads\Prathamesh\VIDTUN.joblib'
# #   joblib.dump(process_video_with_esrgan, path)
# #   process_video_with_esrgan = io.BytesIO(process_video_with_esrgan())


# # VIDTUNE = process_video_with_esrgan()
# # save_process_video_with_esrgan_model(VIDTUNE)