
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


def warp_image_cv(img, c_src, c_dst, dshape = None):
    img = torch.tensor(img).to(device).permute(2,0,1)[None, ...].float()
    dshape = dshape or img.shape
    theta = tps_np.tps_theta_from_points(c_src, c_dst, reduced=True, lambd=0.01)
    theta = torch.tensor(theta).to(device)[None, ...]
    grid = tps_pth.tps_grid(theta, torch.tensor(c_dst, device=device), dshape)
    #print(grid.shape, grid.dtype)
    img = F.grid_sample(img, grid, align_corners=False)
    return img[0].permute(1,2,0).cpu().numpy().astype(np.uint8)


if __name__ == "__main__":

    folder = '/home/avinash/Desktop/datasets/endo/depth/rectified04/cropped/image01/'

    images = sorted(glob.glob(folder + '*.jpg'))

    img1 = cv2.imread(images[0])
    kps1, descs1 = dalf.detectAndCompute(img1)
    nframe = 0

    in_frames = []

    for img in images[1:]:
        in_frames.append(cv2.imread(img))

    out_frames = []

    for img2 in tqdm.tqdm(in_frames[:1000]):
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

        h, w = img1.shape[:2]

        c_src = np.float32([kps1[m.queryIdx].pt for m in good_matches]) / np.float32([w,h])
        c_dst = np.float32([kps2[m.trainIdx].pt for m in good_matches]) / np.float32([w,h])

        #Warp deformed image (img2) into template
        warped = warp_image_cv(img2, c_dst, c_src)

        result = np.hstack([cv2.resize(img1, (w // 4, h // 4)),
                            cv2.resize(img2, (w // 4, h // 4)),
                            cv2.resize(warped, (w // 4, h // 4))])
        out_frames.append(result)

        #plt.imshow(result), plt.show()

    with imageio.get_writer("heart.gif", mode="I") as writer:
    # Loop through the frames and write them to the GIF writer object
        for frame in tqdm.tqdm(out_frames):
            writer.append_data(frame[..., ::-1])

