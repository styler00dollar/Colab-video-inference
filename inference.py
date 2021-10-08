from types import FrameType
from PIL import Image
import argparse
import torch
import torchvision.transforms as TF
import torch.nn as nn
import os
import numpy as np
import cv2
import warnings
import numpy
from tqdm import tqdm
import glob
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

input_video = "a.webm"
os.system("cd /home/x/test/")
os.system("sudo rm -rf /home/x/test/png/")
os.system("mkdir /home/x/test/png/")
os.system("cd /home/x/test/")
os.system(f"ffmpeg -i {input_video} \"png/%05d.png\"")

frames_dir = "png/" #@param
files = sorted(glob.glob(frames_dir + '/**/*.png', recursive=True))
del files[-1]

model_path = "Checkpoint_4_55000_G.pt"
# model, load it manually
#model = 
#model.load_state_dict(torch.load(model_path))

# load a jit
model = torch.jit.load(model_path)
model.cuda().eval()

input_frame = 1
for f in tqdm(files):
  with torch.no_grad():
    filename_frame_1 = f
    filename_frame_2 = os.path.join(frames_dir, f'{input_frame+1:0>5d}.png')
    output_frame_file_path = os.path.join(frames_dir, f"{input_frame:0>5d}_0.5.png")

    img1 = cv2.imread(filename_frame_1)
    img2 = cv2.imread(filename_frame_2)


    # resize input
    img1 = cv2.resize(img1, (1280, 720)) #, interpolation=cv2.INTER_NEAREST)
    img2 = cv2.resize(img2, (1280, 720)) #, interpolation=cv2.INTER_NEAREST)

    img1_new = cv2.resize(img1, (1280, 720)) #, interpolation=cv2.INTER_NEAREST)
    img2_new = cv2.resize(img2, (1280, 720)) #, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename_frame_1, img1_new)
    cv2.imwrite(filename_frame_2, img2_new)

    #img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YUV)
    #img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YUV)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    img1 = torch.from_numpy(img1).unsqueeze(0).permute(0,3,1,2)/255
    img2 = torch.from_numpy(img2).unsqueeze(0).permute(0,3,1,2)/255

    out = model(img1.cuda().contiguous(), img2.cuda().contiguous())

    # to numpy and save
    out = out.data.mul(255).mul(255 / 255).clamp(0, 255).round()
    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy() #*255
    out = out.astype(np.uint8)
    #out = cv2.cvtColor(out, cv2.COLOR_YUV2RGB)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    out = cv2.resize(out, (1280,720)) #, interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(output_frame_file_path, out)

    input_frame += 1

os.system("cd /home/x/test/png/")
os.system(f"ffmpeg -y -r 48 -f image2 -pattern_type glob -i '/home/x/test/png/*.png' -crf 18 /home/x/out.mp4")
os.system("cd /home/x/test/")