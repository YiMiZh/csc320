## CSC320 Winter 2019
## Assignment 2
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

import numpy as np
import cv2 as cv

# File psi.py define the psi class. You will need to
# take a close look at the methods provided in this class
# as they will be needed for your implementation
import psi

# File copyutils.py contains a set of utility functions
# for copying into an array the image pixels contained in
# a patch. These utilities may make your code a lot simpler
# to write, without having to loop over individual image pixels, etc.
import copyutils

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################

# If you need to import any additional packages
# place them here. Note that the reference
# implementation does not use any such packages

#########################################


#########################################
#
# Computing the Patch Confidence C(p)
#
# Input arguments:
#    psiHatP:
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    confidenceImage:
#         An OpenCV image of type uint8 that contains a confidence
#         value for every pixel in image I whose color is already known.
#         Instead of storing confidences as floats in the range [0,1],
#         you should assume confidences are represented as variables of type
#         uint8, taking values between 0 and 255.
#
# Return value:
#         A scalar containing the confidence computed for the patch center
#

def computeC(psiHatP=None, filledImage=None, confidenceImage=None):
    assert confidenceImage is not None
    assert filledImage is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################

    # Replace this dummy value with your own code
    # C = 1
    # get confidence image patch
    c_patch, _ = copyutils.getWindow(confidenceImage, (psiHatP.row(), psiHatP.col()), psiHatP.radius())
    # get filled image patch and it's validation pixels.
    fill_p, valid_p = copyutils.getWindow(filledImage, (psiHatP.row(), psiHatP.col()), psiHatP.radius())
    num_pixel = np.sum(valid_p)
    C = np.sum(c_patch[fill_p > 0]) / num_pixel
    #########################################

    return C

#########################################
#
# Computing the max Gradient of a patch on the fill front
#
# Input arguments:
#    psiHatP:
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    inpaintedImage:
#         A color OpenCV image of type uint8 that contains the
#         image I, ie. the image being inpainted
#
# Return values:
#         Dy: The component of the gradient that lies along the
#             y axis (ie. the vertical axis).
#         Dx: The component of the gradient that lies along the
#             x axis (ie. the horizontal axis).
#

def computeGradient(psiHatP=None, inpaintedImage=None, filledImage=None):
    assert inpaintedImage is not None
    assert filledImage is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################

    # Replace these dummy values with your own code
    # Dy = 1
    # Dx = 0
    # get pixels in psiHatP and the binary array that is zero at all invalid patch pixels.
    pixels, valid_p = psiHatP.pixels(True)
    # get pixels that were not masked in the original image.
    filled = psiHatP.filled()
    # convert the patch to greyscale
    g_patch = cv.cvtColor(pixels, cv.COLOR_BGR2GRAY)
    # I use Sobel operator here to compute dx and dy
    dx = cv.Sobel(g_patch, cv.CV_64F, 1, 0, ksize=5)
    dy = cv.Sobel(g_patch, cv.CV_64F, 0, 1, ksize=5)
    # get gradient
    grad = np.sqrt(np.multiply(dx, dx) + np.multiply(dy, dy))
    # print pixels, grad, valid_p, filled
    # print pixels.shape, grad.shape, valid_p.shape, filled.shape
    # remove invalid pixels use valid_p
    grad = np.multiply(grad, valid_p)
    # remove pixels that has already been filled by using filled
    grad = np.multiply(grad, (filled == 255))
    # find index of argmax of those gradients
    i,j = np.unravel_index(grad.argmax(), grad.shape)
    # get max gradient
    Dx = dx[i][j]
    Dy = dy[i][j]
    #########################################

    return Dy, Dx

#########################################
#
# Computing the normal to the fill front at the patch center
#
# Input arguments:
#    psiHatP:
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    fillFront:
#         An OpenCV image of type uint8 that whose intensity is 255
#         for all pixels that are currently on the fill front and 0
#         at all other pixels
#
# Return values:
#         Ny: The component of the normal that lies along the
#             y axis (ie. the vertical axis).
#         Nx: The component of the normal that lies along the
#             x axis (ie. the horizontal axis).
#
# Note: if the fill front consists of exactly one pixel (ie. the
#       pixel at the patch center), the fill front is degenerate
#       and has no well-defined normal. In that case, you should
#       set Nx=None and Ny=None
#

def computeNormal(psiHatP=None, filledImage=None, fillFront=None):
    assert filledImage is not None
    assert fillFront is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################
    # Replace these dummy values with your own code
    # Ny = 0
    # Nx = 1
    # according to note
    if np.sum(fillFront == 255) == 1:
        Ny, Nx = None, None
    else:
        # fill_p, _ = copyutils.getWindow(filledImage, (psiHatP.row(), psiHatP.col()), psiHatP.radius())
        fill_p, _ = copyutils.getWindow(fillFront, (psiHatP.row(), psiHatP.col()), psiHatP.radius())
        # print fill_p == psiHatP.filled()
        # print fill_p, psiHatP.filled()
        # get dx and dy on point p (center of this patch)
        dx = cv.Sobel(fill_p, cv.CV_64F, 1, 0, ksize=5)[psiHatP.radius()][psiHatP.radius()]
        dy = cv.Sobel(fill_p, cv.CV_64F, 0, 1, ksize=5)[psiHatP.radius()][psiHatP.radius()]
        # get unit vector
        uv = np.sqrt(np.multiply(dx, dx) + np.multiply(dy, dy))
        if uv == 0:
            Ny, Nx = None, None
        else:
            Nx, Ny = dx/uv, -dy/uv
        # Nx = dx
        # Ny = dy
        #########################################

    return Ny, Nx