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
    def sticher(self, img_lst_3):
        #TODO: Take in a 3 length list of images
        #[] = map rgb to gray
        #Loop through Image objects
        #Loading Images and turning to RGB
        keypoints_descriptors=[]
        load_rgb2g = lambda x: color.rgb2gray(io.imread(x))
        gray_img = list(map(load_rgb2g, img_lst_3))


        #ORB is used to find featurs in images, n_keypoints
        #Is the number of features to be found in the Image
        #We use the orb instance to calculate features in each Image
        #We then Find the oevrlaping Features and remove the Redundant features
        orb = ORB(n_keypoints=800, fast_threshold= 0.07)
        for img in gray_img:
            orb.detect_and_extract(img)
            keypoints_descriptors.append((orb.keypoints, orb.descriptors))
        
        matches01 = match_descriptors(keypoints_descriptors[0][1], keypoints_descriptors[1][1], cross_check=True)
        matches12 = match_descriptors(keypoints_descriptors[1][1], keypoints_descriptors[2][1], cross_check=True)
        #We use the RANSAC algorithm to remove the extra features
        src = keypoints_descriptors[0][0][matches01[:, 0]][:, ::-1]
        dst = keypoints_descriptors[1][0][matches01[:, 1]][:, ::-1]

        model_robust01, inliers01 = ransac((src, dst), ProjectiveTransform, min_samples=4, residual_threshold=1, max_trials=300)

        src = keypoints_descriptors[2][0][matches12[:, 1]][:, ::-1]
        dst = keypoints_descriptors[1][0][matches12[:, 0]][:, ::-1]

        model_robust12, inliers12 = ransac((src, dst), ProjectiveTransform, min_samples=4, residual_threshold=1, max_trials=300)

        #Warping the Images to stitch together
        r, c = gray_img[0].shape[:2]
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
        image1_warped = warp(gray_img[0], transform01, order=3,output_shape=output_shape, cval=-1)

        image1_mask = (image1_warped != -1)  # Mask == 1 inside image
        image1_warped[~image1_mask] = 0      # Return background values to 0


        image2_warped = warp(gray_img[1], offset1.inverse, order=3, output_shape=output_shape, cval=-1)

        image2_mask = (image2_warped != -1)  # Mask == 1 inside image
        image2_warped[~image2_mask] = 0      # Return background values to 0

        transform12 = (model_robust12 + offset1).inverse
        image3_warped = warp(gray_img[2], transform12, order=3, output_shape=output_shape, cval=-1)


        image3_mask = (image3_warped != -1)  # Mask == 1 inside image
        image3_warped[~image3_mask] = 0

        merged = (image1_warped + image2_warped + image3_warped)
        overlap = (image1_mask* 1.0 + image2_mask + image3_mask)
        normalized = merged / np.maximum(overlap, 1)
        return normalized
