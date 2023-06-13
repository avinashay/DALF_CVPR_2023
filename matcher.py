
import torch
import cv2
import torch.nn.functional as F
import imageio
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
import glob


from modules.models.DALF import DALF_extractor as DALF
from modules.tps import RANSAC
from modules.tps import numpy as tps_np
from modules.tps import pytorch as tps_pth


device = 'cuda'

dalf = DALF(dev = device)


if __name__ == "__main__":

    folder = '/home/avinash/Desktop/datasets/endo/depth/rectified01/cropped/image01/'

    images = sorted(glob.glob(folder + '*.jpg'))

    img1 = cv2.imread(images[0])
    kps1, descs1 = dalf.detectAndCompute(img1)

    in_frames = []

    for img in images[1:]:
        in_frames.append(cv2.imread(img))

    out_frames = []

    frame_idx = 0

    for i in tqdm.tqdm(range(1,len(in_frames))):

        img2 = in_frames[i]

        #Compute DALF features
        kps2, descs2 = dalf.detectAndCompute(img2)

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

        frame_idx += 1

        if frame_idx % 10 == 0:
            img1 = img2
            kps1 = kps2
            descs1 = descs2

        c_src = np.int32([kps1[m.queryIdx].pt for m in good_matches])
        c_dst = np.int32([kps2[m.trainIdx].pt for m in good_matches]) 



        


        cv2.imshow('Result', result)
        k = cv2.waitKey(1)

        if k == ord('q'):
            break




