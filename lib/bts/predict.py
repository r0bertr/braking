import os
import sys

import bts
import cv2
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torchvision.transforms
from tqdm import tqdm


class Args(dict):
    """
    Copy from https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    Example:
    m = Args({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Args, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Args, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Args, self).__delitem__(key)
        del self.__dict__[key]

if len(sys.argv) != 5 and len(sys.argv) != 4:
    print('Usage: python predict.py <path_to_model> <path_to_seq> <path_to_output_dir> [calib_date]')
    exit(1)

model_path = sys.argv[1]
image_dir = sys.argv[2]
output_dir = sys.argv[3]
calib_date = ''
if len(sys.argv) == 5:
    calib_date = sys.argv[4]

os.makedirs(output_dir, exist_ok=True)

# ================ Parameters ======================
NUSCENES_DATAROOT = '/home/user/bts/data/ITS/nuScenes'
params = Args()
if image_dir[-3:] == 'txt':
    params.dataset = 'nuScenes'
else:
    params.dataset = 'kitti'
if calib_date == '2011_09_30':
    params.focal = 7.070912e+02
    params.input_width = 1226
    params.input_height = 370
elif calib_date == '2011_10_03':
    params.focal = 7.188560e+02
    params.input_width = 1241
    params.input_height = 376
elif calib_date == '2011_09_26':
    params.focal = 721.5377
    params.input_width = 1242
    params.input_height = 375
elif calib_date == '2011_09_29':
    params.focal = 718.3351
    params.input_width = 1238
    params.input_height = 374
elif calib_date == '2011_09_28':
    params.focal = 707.0493
    params.input_width = 1224
    params.input_height = 370
elif params.dataset == 'kitti':
    print('Unsupported calib_date. Abort.')
    exit(1)

if params.dataset == 'nuScenes':
    nusc = NuScenes(version='v1.0-mini', dataroot=NUSCENES_DATAROOT, verbose=False)
    scene_token = image_dir.split('/')[-2]
    first_sample_token = nusc.get('scene', scene_token)['first_sample_token']
    sample_data_token = nusc.get('sample', first_sample_token)['data']['CAM_FRONT']
    calibrated_sensor_token = nusc.get('sample_data', sample_data_token)['calibrated_sensor_token']
    intrinsic = nusc.get('calibrated_sensor', calibrated_sensor_token)['camera_intrinsic']
    params.focal = intrinsic[0][0]
    params.input_width = 1600
    params.input_height = 900

params.encoder = 'densenet161_bts'
params.model_name = 'bts_eigen_v2_pytorch_densenet161'
# bts_size is the initial num_filters in bts. Default = 512.
params.bts_size = 512
params.max_depth = 80
params.training_focal = 715.0873
# ==================================================

if params.dataset == 'nuScenes':
    with open(image_dir) as fp:
        image_paths = fp.readlines()
    image_paths = [os.path.join(NUSCENES_DATAROOT, x.rstrip()) for x in image_paths]
    image_names = [os.path.basename(x) for x in image_paths]
else:
    image_names = sorted(os.listdir(image_dir))
    image_paths = [os.path.join(image_dir, x) for x in image_names]

model = bts.BtsModel(params=params)
model = torch.nn.DataParallel(model)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model'])
model.eval()
model.cuda()

with torch.no_grad():
    print('Predicting {}...'.format(image_dir))
    for idx, image_path in enumerate(tqdm(image_paths)):
        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0
        # Scaling the image such that width & 32 == 0 && height & 32 == 0
        # height = image.shape[0]
        # width = image.shape[1]
        # resized_width = int(round(width / 32) * 32)
        # resized_height = int(round(height / 32) * 32)
        # image = cv2.resize(image, (resized_width, resized_height))
        # Crop the image such that width & 32 == 0 && height & 32 == 0
        height = image.shape[0]
        width = image.shape[1]
        cropped_width = width // 32 * 32
        left = (width - cropped_width) // 2
        cropped_height = height // 32 * 32
        top = (height - cropped_height) // 2
        image = image[top:top+cropped_height, left:left+cropped_width]
        # Create the tensor and normalize.
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        image = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        # Expand the first dim.
        image = image[None, :]
        # Convert to Variables.
        image = torch.autograd.Variable(image.cuda())
        focal = torch.autograd.Variable(torch.tensor([params.focal], dtype=torch.float64).cuda())
        # Predict!
        _, _, _, _, depth_est = model(image, focal)
        # Save the depth image.
        depth_map = np.squeeze(depth_est.detach().cpu().numpy(), axis=(0, 1))
        # Scaling the image back to the original width/height.
        # depth_map = cv2.resize(depth_map, (width, height))
        # Padding the image back to the original width/height
        depth_map = np.pad(depth_map, ((top, height - top - cropped_height), (left, width - left - cropped_width)), 'constant', constant_values=(0))
        # Rescale the estimated depth with focal length
        depth_map = depth_map * (params.focal / params.training_focal)
        cv2.imwrite(os.path.join(output_dir, os.path.splitext(image_names[idx])[0] + '.tiff'), depth_map)
