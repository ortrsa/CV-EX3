import sys
from typing import List
import numpy as np
import cv2
from numpy.linalg import LinAlgError


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size:
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    # normalize image if needed
    if im1.max() > 1:
        im1 /= 255
        im2 /= 255

    point = []
    direct = []

    h_size = im1.shape[0]
    w_size = im1.shape[1]

    # use sobel to compute the gradients for Ix and Iy
    Ix = cv2.Sobel(im1, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(im1, cv2.CV_64F, 0, 1, ksize=3)
    It = im2 - im1

    # make windows in size "win_size" and iterate in "step_size".
    for i in range(0, h_size - win_size, step_size):
        for j in range(0, w_size - win_size, step_size):
            cropIx = Ix[i:i + win_size, j:j + win_size]
            cropIy = Iy[i:i + win_size, j:j + win_size]
            cropIt = It[i:i + win_size, j:j + win_size]

            # make matrix A and vector b
            A = np.array([cropIx.flatten(), cropIy.flatten()]).T
            b = cropIt.flatten().T

            # get eigen values to filter non relevant "arrows"
            eig1, eig2 = np.linalg.eigvals(np.dot(A.T, A))
            eigb = max(eig1, eig2) * 255
            eigs = min(eig1, eig2) * 255

            if ((eigb >= eigs) and (eigs > 5)) and eigb / eigs < 50:
                u = np.dot(np.linalg.pinv(A), -b)
                point.append([j, i])
                direct.append([u[0], u[1]])

    point = np.array(point)
    direct = np.array(direct)
    return point, direct


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """

    lap = []
    exp_list = []
    kernel = cv2.getGaussianKernel(5, -1)
    g_pyr = gaussianPyr(img, levels)
    for i in range(1, len(g_pyr)):
        exp_list.append(gaussExpand(g_pyr[i], kernel))
        # crop expanded image if needed
        if g_pyr[i - 1].shape != exp_list[i - 1].shape:
            exim = exp_list[i - 1]
            exim = exim[0:g_pyr[i - 1].shape[0], 0:g_pyr[i - 1].shape[1]]
            exp_list[i - 1] = exim
        lap.append(g_pyr[i - 1] - exp_list[i - 1])
    lap.append(g_pyr[-1])
    return lap


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Resotrs the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    kernel = cv2.getGaussianKernel(5, -1)
    lap_copy = lap_pyr.copy()
    lap_copy.reverse()
    image = lap_copy[0]
    for i in range(1, len(lap_copy)):
        expandedImg = gaussExpand(image, kernel)
        if lap_copy[i].shape != expandedImg.shape:
            expandedImg = expandedImg[0:lap_copy[i].shape[0], 0:lap_copy[i].shape[1]]
        image = expandedImg + lap_copy[i]
    return image


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """

    pyr = [img]
    # ksize - kernel size, should be odd and positive (3,5,...)
    # sigma - Gaussian standard deviation.
    # If it is non-positive, it is computed from ksize as
    # sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    gk = cv2.getGaussianKernel(5, -1)
    kernel = np.dot(gk, gk.T)
    for i in range(levels):
        img = cv2.filter2D(pyr[i], -1, kernel, borderType=cv2.BORDER_REPLICATE)
        # take every second pixel and add the new image to list
        img = img[::2, ::2]
        pyr.append(img)
    return pyr


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    # if kernel is 1D
    if gs_k.shape[1] == 1 and gs_k.shape[0] != 1:
        gs_k = np.dot(gs_k, gs_k.T)

    h_size = img.shape[0] * 2
    w_size = img.shape[1] * 2
    # make zeros matrix for gray or colored image
    if len(img.shape) > 2:
        new_im = np.zeros((h_size, w_size, 3))
    else:
        new_im = np.zeros((h_size, w_size))
    # equals every second pix to the small image and conv with gs_k
    new_im[::2, ::2] = img
    new_im = (cv2.filter2D(new_im, -1, gs_k * 4))
    return new_im


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: Blended Image
    """
    # make the image same shape
    sml_imPadded = imagePadding(img_1, img_2)
    mask_Padded = imagePadding(img_1, mask)

    # make gaussianPyr from mask and laplaceian from img 1 and 2.
    Gm = gaussianPyr(mask_Padded, levels)
    La = laplaceianReduce(img_1, levels)
    Lb = laplaceianReduce(sml_imPadded, levels)
    Lc = []
    for i in range(len(Gm)):
        Lc.append(Lb[i] * Gm[i] + (1 - Gm[i]) * La[i])
    # expand the image's and mask together.
    BlendedImage = laplaceianExpand(Lc)
    Naiveblend = sml_imPadded * mask_Padded + (1 - mask_Padded) * img_1

    return Naiveblend, BlendedImage


def imagePadding(image_1, image_2):
    """
    padding the small image with zero's to fit the big image shape.
    :param image_1:
    :param image_2:
    :return:
    """

    if image_1.shape >= image_2.shape:
        big_im = image_1
        sml_im = image_2
    else:
        big_im = image_2
        sml_im = image_1

    sml_imPadd = np.zeros(big_im.shape)
    sml_imPadd[(big_im.shape[0] - sml_im.shape[0]) // 2:(big_im.shape[0] - sml_im.shape[0]) // 2 + sml_im.shape[0],
    (big_im.shape[1] - sml_im.shape[1]) // 2:(big_im.shape[1] - sml_im.shape[1]) // 2 + sml_im.shape[1]] = sml_im
    return sml_imPadd
