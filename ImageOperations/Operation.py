import cv2
import numpy as np
import matplotlib.pyplot as plt
class ImageOp():
    def __init__(self):
        pass
    def stitch(self, img1, img2):
        img_ = cv2.imread(img2)
        img = cv2.imread(img1)
        img1 = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        good = []
        for m in matches:
            if m[0].distance < 0.5*m[1].distance:
                good.append(m)
                matches = np.asarray(good)

        if len(matches[:,0]) >= 4:
            src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
            dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,1,2)
            H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        else:
            pass
        dst = cv2.warpPerspective(img_,H,(img.shape[1] + img_.shape[1], img.shape[0]))
        plt.subplot(122),plt.imshow(dst)
        dst[0:img.shape[0], 0:img.shape[1]] = img
        cv2.imwrite("output.jpg",dst)

