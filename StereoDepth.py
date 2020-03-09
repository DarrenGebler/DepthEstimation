import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from sklearn.preprocessing import normalize
import imageio


def process_frame(left, right, name):
    kernel_size = 3
    smooth_left = cv2.GaussianBlur(left, (3, 3), 1.5)
    smooth_right = cv2.GaussianBlur(right, (3, 3), 1.5)

    window_size = 9
    left_matcher = cv2.StereoSGBM_create(numDisparities=96, blockSize=7, P1=8 * 3 * 9 ** 2, P2=32 * 3 * 9 ** 2,
                                         disp12MaxDiff=1,
                                         uniquenessRatio=16, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(80000)
    wls_filter.setSigmaColor(1.2)

    disparity_left = np.int16(left_matcher.compute(smooth_left, smooth_right))
    disparity_right = np.int16(right_matcher.compute(smooth_right, smooth_left))

    wls_image = wls_filter.filter(disparity_left, smooth_left, None, disparity_right)
    wls_image = cv2.normalize(src=wls_image, dst=wls_image, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    wls_image = np.uint8(wls_image)

    fig = plt.figure(figsize=(wls_image.shape[1] / 96, wls_image.shape[0] / 96), dpi=96, frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(wls_image, cmap='jet')
    plt.savefig("input/1/disparities/")
    plt.close()
    create_combined_output(left, right, name)


def create_combined_output(left, right, name):
    combined = np.concatenate((left, right, cv2.imread("input/1/disparities/" + name)), axis=0)
    cv2.imwrite("input/1/combined/" + name, combined)


def process_dataset():
    left_images = [x for x in os.listdir("input/1/left/") if not x.startswith('.')]
    right_images = [x for x in os.listdir("input/1/right/") if not x.startswith('.')]
    assert (len(left_images) == len(right_images))
    left_images.sort()
    right_images.sort()
    for i in range(len(left_images)):
        left_image_path = "input/1/left/" + left_images[i]
        right_image_path = "input/1/right/" + right_images[i]
        left_image = cv2.imread(left_image_path, cv2.IMREAD_COLOR)
        right_image = cv2.imread(right_image_path, cv2.IMREAD_COLOR)
        process_frame(left_image, right_image, left_images[i])


if __name__ == '__main__':
    process_dataset()
    images = []
    for filename in os.listdir("input/1/combined/"):
        images.append(imageio.imread("input/1/combined/" + filename))
    print("final")
    imageio.mimsave('input/1/result.gif', images)
