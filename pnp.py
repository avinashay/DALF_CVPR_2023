
import torch
import cv2
import torch.nn.functional as F
import imageio
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
import glob
import os
import open3d as o3d
import time 
from modules.models.DALF import DALF_extractor as DALF
from modules.tps import RANSAC
from modules.tps import numpy as tps_np
from modules.tps import pytorch as tps_pth


device = 'cuda'

dalf = DALF(dev = device)

def load_data(folder):

    img_folder = folder+"/cropped/image01/"
    depth_folder = folder+"/cropped/depth01/"

    intrinscis = os.path.join(folder,"cropped/intrinscis.txt")
    ext = os.path.join(folder,"cropped/extrinsics.txt")

    K = np.loadtxt(intrinscis)

    print("intrinsic = " , K)

    stereo = np.loadtxt(ext)

    b = stereo[0,3]/1000

    print("basline = ", b)

    limages = sorted(glob.glob(img_folder+"*.jpg"))[::10]
    idepths = sorted(glob.glob(depth_folder+"*.npy"))[::10]

    return K, b , limages, idepths

def inv_project(depths, K):
    """ Pinhole camera inverse-projection """

    ht, wd = depths.shape

    fx, fy, cx, cy = K[0,0],K[1,1],K[0,2],K[1,2]

    x, y = np.meshgrid(
        np.arange(wd).astype(np.float32),
        np.arange(ht).astype(np.float32))
    
    depths = depths

    X = depths * ((x - cx) / fx)
    Y = depths * ((y - cy) / fy)
    Z = depths

    return np.stack([X, Y, Z], axis=-1)

def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """

    CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

    CAM_LINES = np.array([
        [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

def viz_open3d(trajectory, images, idepths, K, b):

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(height=540, width=960)

    img1 = cv2.imread(images[0])
    last_depth = np.load(idepths[0])
    last_depth = (K[0,0]*-b) / (last_depth)
    xyz1 = inv_project(last_depth,K)
    xyz = xyz1.reshape(-1,3)
    clr = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    clr = clr.reshape(-1,3)/255
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(clr)
    vis.add_geometry(pcd)
    cam = create_camera_actor(g=1,scale=0.005)
    cam.transform(np.eye(4))
    vis.add_geometry(cam)

    ti = np.eye(4)


    for i in range(1,len(trajectory)):

        if i==len(trajectory)-1:
            cam = create_camera_actor(g=0.4,scale=0.001)
        else:
            cam = create_camera_actor(g=0,scale=0.001)
        ti = np.matmul(trajectory[i],ti)
        cam.transform(np.linalg.inv(ti))
        vis.add_geometry(cam)

    
    vis.run()
    vis.destroy_window()



if __name__ == "__main__":

    folder = '/home/avinash/Desktop/datasets/endo/depth/rectified04/'

    K, b, images, idepths = load_data(folder)
    b = -b

    traj = []

    for i in tqdm.tqdm(range(len(images)-1)):

        img1 = cv2.imread(images[i])
        kps1, descs1 = dalf.detectAndCompute(img1)

        img2 = cv2.imread(images[i+1])
        kps2, descs2 = dalf.detectAndCompute(img2)

        last_depth = np.load(idepths[i])
        last_depth = (K[0,0]*-b) / (last_depth)
        xyz1 = inv_project(last_depth,K)


        #Match using vanilla opencv matcher
        matcher = cv2.BFMatcher(crossCheck = True)
        matches = matcher.match(descs1, descs2)

        src_pts = np.float32([kps1[m.queryIdx].pt for m in matches])
        tgt_pts = np.float32([kps2[m.trainIdx].pt for m in matches])

        #Computes non-rigid RANSAC
        inliers = RANSAC.nr_RANSAC(src_pts, tgt_pts, device,  batch = 3_000, thr = 0.2)
        good_matches = [matches[i] for i in range(len(matches)) if inliers[i]]

        best_matches = sorted(good_matches, key = lambda x:x.distance)[:250]

        result = cv2.drawMatches(img1, kps1, img2, kps2, best_matches, None, matchColor = (0,255,0), matchesMask = None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


        cv2.imshow('Result', result)
        k = cv2.waitKey(1)

        if k == ord('q'):
            break

        src_pts = np.int32([kps1[m.queryIdx].pt for m in best_matches])
        tgt_pts = np.float32([kps2[m.trainIdx].pt for m in best_matches])

        object_points = []
        images_points = []

        for (x0, y0), (x1, y1), i in zip(src_pts, tgt_pts, range(len(src_pts))):
                object_points.append(xyz1[y0,x0,:])
                images_points.append([x1,y1])
        

        object_points = np.array(object_points).reshape(-1,3)
        images_points = np.array(images_points).reshape(-1,2)

        rec,rvec,tvec,_ = cv2.solvePnPRansac(object_points,images_points,K, distCoeffs=np.array([0.0,0.0,0.0,0.0]), iterationsCount=100,reprojectionError=10.0, confidence=0.80)

        if rec == True:
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()
            traj.append(T)

        torch.cuda.empty_cache()

    viz_open3d(traj, images, idepths, K, b)



      


