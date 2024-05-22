import glob

import numpy as np
import matplotlib.pyplot as plt
import cv2

from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage

# compute the intrisinic matrix
fx, fy, cx, cy, _, LUT = ReadCameraModel('Oxford_dataset_reduced/model/')
K = np.array([
  [fx, 0, cx],
  [0, fy, cy],
  [0, 0, 1]
])

# load and demosaic images
imgs = []
for path in sorted(glob.glob('Oxford_dataset_reduced/images/*')):
  img = cv2.imread(path, flags=-1)
  color_image = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
  undistorted = UndistortImage(color_image, LUT)
  imgs.append(color_image)

imgs = np.array(imgs)

# find keypoints correspondances
pos = np.reshape(np.eye(4), (1, 4, 4))

for i in range(imgs.shape[0] - 1):
  img1 = imgs[i]
  img2 = imgs[i + 1]
  
  # use SIFT to find keypoints and descriptors
  sift = cv2.SIFT_create()
  kp1, des1 = sift.detectAndCompute(img1, None)
  kp2, des2 = sift.detectAndCompute(img2, None)

  flann = cv2.FlannBasedMatcher()
  possible = flann.knnMatch(des1, des2, k=2)
  
  # filter out bad matches
  ratio = 0.5
  matches = []
  for m, n in possible:
    if m.distance < ratio * n.distance:
      matches.append(m)

  # estimate fundamental matrix
  img1_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
  img2_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
  
  F, mask = cv2.findFundamentalMat(img1_pts, img2_pts, cv2.FM_RANSAC)

  # recover essential matrix
  E = K.T @ F @ K

  # reconstruct rotation and translation parameters
  _, R, t, _ = cv2.recoverPose(E, img1_pts, img2_pts, K, mask=mask)

  # create transformation matrix
  T = np.empty((4, 4))
  T[:3, :3] = R
  T[:3, 3] = np.reshape(t, 3)
  T[3] = [0, 0, 0, 1]

  # update position in array
  pos = np.concatenate((pos, np.expand_dims(pos[-1] @ T, axis=0)))

# create 2d map of position
plt.figure(figsize=(6, 6))
plt.scatter(pos[:, 0, 3], pos[:, 2, 3])
plt.xlabel('X')
plt.ylabel('Z')
plt.title('Position in 2D')
plt.show()

# create 3d map of position
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos[:, 0, 3], pos[:, 2, 3], pos[:, 1, 3])
ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_title('Position in 3D')
plt.show()


