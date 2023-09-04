from collections import OrderedDict
import torch
import torch.nn as nn

import math

import sys
import os.path
import glob
import cv2
import numpy as np 

import torchvision
from Architecture import RRDB_Net
import matplotlib.pyplot as plt
import torch.optim as optim




# folder of test videos
test_vid_folder = '/kaggle/input/videos-for-esr/video/test360/*'

# initialize pre-trained ESRGAN fine tuned for videos
model_path = "/kaggle/input/esrgan-model/VID_tune"
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
model = RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', \
                        mode='CNA', res_scale=1, upsample_mode='upconv')
model.load_state_dict(torch.load(model_path), strict=True)

# switch to evaluate mode
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to(device)
print('Model path {:s}. \nProcessing Video...'.format(model_path))

# iterate through all videos in test folder
for path in glob.glob(test_vid_folder):

    # start video capture
    cap = cv2.VideoCapture(path)

    # Define the codec and create VideoWriter object
    base = os.path.splitext(os.path.basename(path))[0]
    FPS = cap.get(cv2.CAP_PROP_FPS)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('/kaggle/working/{:s}_ESRGAN.avi'.format(base),
                          fourcc,
                          cap.get(cv2.CAP_PROP_FPS),
                          (int(width * 4), int(height * 4)))

    # process video
    while(cap.isOpened()):

        # read a frame of the video
        ret, img = cap.read()
        if ret == True:

            # pre-process frame to expected model input format
            img = img * 1.0 / 255
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img_LR = img.unsqueeze(0)
            img_LR = img_LR.to(device)

            # generate a super resolution frame
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)

            # write the super resolution frame frame
            out.write(output)

           
            # Display the super resolution frame using plt.imshow
            
        else:
            break
            

    # Release everything if job is finished
    cap.release()
    out.release()
    plt.close()
    #cv2.destroyAllWindows()