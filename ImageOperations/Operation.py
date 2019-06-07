import cv2
import numpy as np
import matplotlib.pyplot as plit
from skimage import io
from skimage import color
from skimage.feature import ORB
from skimage.feature import match_descriptors
from skimage.transform import ProjectiveTransform
from skimage.measure import ransac
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.feature import match_descriptors


#TODO:  Write repeating lines in a map
#TODO:  Write the function Stitch to work for n number of images

class Stitch():
    def __init__(self):
          pass
    def sticher(self, img1, img2, img3):
        #Loading Images and turning to RGB
        image1 = color.rgb2gray(io.imread(img1))
        image2 = color.rgb2gray(io.imread(img2))
        image3 = color.rgb2grey(io.imread(img3))

        #ORB is used to find featurs in images, n_keypoints
        #Is the number of features to be found in the Image
        #We use the orb instance to calculate features in each Image
        #We then Find the oevrlaping Features and remove the Redundant features
        orb = ORB(n_keypoints=800, fast_threshold= 0.07)
        orb.detect_and_extract(image1)
        keypoints0 = orb.keypoints
        descriptors0 = orb.descriptors

        orb.detect_and_extract(image2)
        keypoints1 = orb.keypoints
        descriptors1 = orb.descriptors

        orb.detect_and_extract(image3)
        keypoints2 = orb.keypoints
        descriptors2 = orb.descriptors
        matches01 = match_descriptors(descriptors0, descriptors1, cross_check=True)
        matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
        #We use the RANSAC algorithm to remove the extra features
        src = keypoints0[matches01[:, 0]][:, ::-1]
        dst = keypoints1[matches01[:, 1]][:, ::-1]

        model_robust01, inliers01 = ransac((src, dst), ProjectiveTransform, min_samples=4, residual_threshold=1, max_trials=300)

        src = keypoints2[matches12[:, 1]][:, ::-1]
        dst = keypoints1[matches12[:, 0]][:, ::-1]

        model_robust12, inliers12 = ransac((src, dst), ProjectiveTransform, min_samples=4, residual_threshold=1, max_trials=300)

        #Warping the Images to stitch together
        r, c = image1.shape[:2]
        corners = np.array([[0, 0],[0, r], [c, 0], [c, r]])
        warped_corners01 = model_robust01(corners)
        warped_corners12 = model_robust12(corners)

        # Find the extents of both the reference image and the warped
        # target image
        all_corners = np.vstack((warped_corners01, warped_corners12, corners))

        # The overally output shape will be max - min
        corner_min = np.min(all_corners, axis=0)
        corner_max = np.max(all_corners, axis=0)
        output_shape = (corner_max - corner_min)

        # Ensure integer shape with np.ceil and dtype conversion
        output_shape = np.ceil(output_shape[::-1]).astype(int)


        offset1 = SimilarityTransform(translation= -corner_min)
        transform01 = (model_robust01 + offset1).inverse
        image1_warped = warp(image1, transform01, order=3,output_shape=output_shape, cval=-1)

        image1_mask = (image1_warped != -1)  # Mask == 1 inside image
        image1_warped[~image1_mask] = 0      # Return background values to 0


        image2_warped = warp(image2, offset1.inverse, order=3, output_shape=output_shape, cval=-1)

        image2_mask = (image2_warped != -1)  # Mask == 1 inside image
        image2_warped[~image2_mask] = 0      # Return background values to 0

        transform12 = (model_robust12 + offset1).inverse
        image3_warped = warp(image3, transform12, order=3, output_shape=output_shape, cval=-1)


        image3_mask = (image3_warped != -1)  # Mask == 1 inside image
        image3_warped[~image3_mask] = 0

        merged = (image1_warped + image2_warped + image3_warped)
        overlap = (image1_mask* 1.0 + image2_mask + image3_mask)
        normalized = merged / np.maximum(overlap, 1)
        return normalized
