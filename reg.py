from modules.models.DALF import DALF_extractor as DALF
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
from modules.tps import RANSAC
from modules.tps import pytorch as tps_pth
from modules.tps import numpy as tps_np
import torch.nn.functional as F



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = 'cpu'
dalf = DALF(dev = device)



img1 = cv2.imread('/home/avinash/Desktop/datasets/endo/depth/rectified01/cropped/image01/0000000000.jpg')
img2 = cv2.imread('/home/avinash/Desktop/datasets/endo/depth/rectified01/cropped/image01/0000000100.jpg')



#display original imgs
plt.figure(figsize = (10,10))
plt.imshow(np.hstack([img1, img2])[..., ::-1]), plt.show()

#Compute kps and features
kps1, descs1 = dalf.detectAndCompute(img1)
kps2, descs2 = dalf.detectAndCompute(img2)

#Match using vanilla opencv matcher
matcher = cv2.BFMatcher(crossCheck = True)
matches = matcher.match(descs1, descs2)

#Draw RAW matches
result = cv2.drawMatches(img1, kps1, img2, kps2, matches, None, matchColor = (0,255,0), matchesMask = None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize = (10,10))
plt.imshow(result[..., ::-1])
plt.show()

src_pts = np.float32([kps1[m.queryIdx].pt for m in matches])
tgt_pts = np.float32([kps2[m.trainIdx].pt for m in matches])

#Computes non-rigid RANSAC
inliers = RANSAC.nr_RANSAC(src_pts, tgt_pts, device, thr = 0.2)

good_matches = [matches[i] for i in range(len(matches)) if inliers[i]]
#Draw filtered matches
result = cv2.drawMatches(img1, kps1, img2, kps2, good_matches, None, matchColor = (0,255,0), matchesMask = None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize = (10,10))
plt.imshow(result[..., ::-1]), plt.show()

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

c_src = np.float32([kps1[m.queryIdx].pt for m in good_matches]) / np.float32([w1,h1])
c_dst = np.float32([kps2[m.trainIdx].pt for m in good_matches]) / np.float32([w2,h2])

def warp_image_cv(img, c_src, c_dst, dshape = None):
    img = torch.tensor(img).to(device).permute(2,0,1)[None, ...].float()
    dshape = dshape or img.shape
    theta = tps_np.tps_theta_from_points(c_src, c_dst, reduced=True, lambd=0.01)
    theta = torch.tensor(theta).to(device)[None, ...]
    grid = tps_pth.tps_grid(theta, torch.tensor(c_dst, device=device), dshape)
    #print(grid.shape, grid.dtype)
    img = F.grid_sample(img, grid)
    return img[0].permute(1,2,0).cpu().numpy().astype(np.uint8)

#Warp deformed image (img2) into template
warped = warp_image_cv(img2, c_dst, c_src)

result = np.hstack([img1, warped])
plt.figure(figsize = (10,10))
plt.imshow(result[..., ::-1]), plt.show()