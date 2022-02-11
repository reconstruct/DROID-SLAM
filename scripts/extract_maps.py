from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / 'droid_slam'))
from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from torch.multiprocessing import Process
import open3d as o3d
import torch.nn.functional as F

from droid import Droid
import droid_backends
from visualization import create_camera_actor, create_point_actor


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--output_ply_file", type=str, help="path to save point cloud")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None
    num_imgs = len(list(Path(args.imagedir).iterdir()))

    tstamps = []
    for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.calib, args.stride), total=num_imgs):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        droid.track(t, image, intrinsics=intrinsics)

    traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))


    # write output to file
    video = droid.video
    dirty_index = torch.where(video.dirty)[0]
    poses = torch.index_select(video.poses, 0, dirty_index)
    disps = torch.index_select(video.disps, 0, dirty_index)
    points = droid_backends.iproj(lietorch.SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()
    images = torch.index_select(video.images, 0, dirty_index)
    images = images.cpu()[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
    thresh = 0.005 * torch.ones_like(disps.mean(dim=[1,2]))

    count = droid_backends.depth_filter(video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)
    count = count.cpu()
    disps = disps.cpu()
    masks = ((count >= 2) & (disps > .5*disps.mean(dim=[1,2], keepdim=True))).reshape(-1)

    points = points.reshape(-1, 3)
    # masks = torch.ones_like(masks.reshape(-1))
    images = images.reshape(-1, 3)

    pts = points[masks].cpu().numpy()
    clr = images[masks].cpu().numpy()

    point_actor = create_point_actor(pts, clr)


    pc = o3d.geometry.PointCloud(point_actor)
    Path(args.output_ply_file).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(args.output_ply_file, pc)
