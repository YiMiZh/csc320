# CSC320 Winter 2019
# Assignment 4
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# import basic packages
import numpy as np
# import the heapq package
from heapq import heappush, heappushpop, nlargest
# see below for a brief comment on the use of tiebreakers in python heaps
from itertools import count

_tiebreaker = count()

from copy import deepcopy as copy

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the Generalized PatchMatch
# algorithm, as explained in Section 3.2 of the PatchMatch paper and Section 3
# of the Generalized PatchMatch paper.
#
# The function takes k NNFs as input, represented as a 2D array of heaps and an
# associated 2D array of dictionaries. It then performs propagation and random search
# as in the original PatchMatch algorithm, and returns an updated 2D array of heaps
# and dictionaries
#
# The function takes several input arguments:
#     - source_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the source image,
#                            as computed by the make_patch_matrix() function. For an
#                            NxM source image and patches of width P, the matrix has
#                            dimensions NxMxCx(P^2) where C is the number of color channels
#                            and P^2 is the total number of pixels in the patch. The
#                            make_patch_matrix() is defined below and is called by the
#                            initialize_algorithm() method of the PatchMatch class. For
#                            your purposes, you may assume that source_patches[i,j,c,:]
#                            gives you the list of intensities for color channel c of
#                            all pixels in the patch centered at pixel [i,j]. Note that patches
#                            that go beyond the image border will contain NaN values for
#                            all patch pixels that fall outside the source image.
#     - target_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the target image.
#     - f_heap:              For an NxM source image, this is an NxM array of heaps. See the
#                            helper functions below for detailed specs for this data structure.
#     - f_coord_dictionary:  For an NxM source image, this is an NxM array of dictionaries. See the
#                            helper functions below for detailed specs for this data structure.
#     - alpha, w:            Algorithm parameters, as explained in Section 3 and Eq.(1)
#     - propagation_enabled: If true, propagation should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step
#     - random_enabled:      If true, random search should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step.
#     - odd_iteration:       True if and only if this is an odd-numbered iteration.
#                            As explained in Section 3.2 of the paper, the algorithm
#                            behaves differently in odd and even iterations and this
#                            parameter controls this behavior.
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure
#     NOTE: the variables f_heap and f_coord_dictionary are modified in situ so they are not
#           explicitly returned as arguments to the function


def propagation_and_random_search_k(source_patches, target_patches,
                                    f_heap,
                                    f_coord_dictionary,
                                    alpha, w,
                                    propagation_enabled, random_enabled,
                                    odd_iteration,
                                    global_vars
                                    ):
    #################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES   ###
    ###  THEN START MODIFYING IT AFTER YOU'VE     ###
    ###  IMPLEMENTED THE 2 HELPER FUNCTIONS BELOW ###
    #################################################
    new_f = f.copy()
    y, x = source_patches.shape[:2]
    # (285, 630, 3, 49) (285, 630, 3, 49) (285, 630, 2)
    # print source_patches.shape, target_patches.shape, f.shape
    # initial our D matrix (should be large at first)
    if best_D is None:
        best_D = np.multiply(np.ones((y, x)), np.inf)
    # replace NaN with zero in both patch
    # source_patches = np.nan_to_num(source_patches)
    # target_patches = np.nan_to_num(target_patches)
    # Firstly see which iteration we are at, and decide how do we loop.
    if odd_iteration:
        # print "odd"
        loop_i = range(y)
        loop_j = range(x)
    else:
        loop_i = range(y - 1, -1, -1)
        loop_j = range(x - 1, -1, -1)
        # See if we want to preform propagation
    for i in loop_i:
        for j in loop_j:
            if propagation_enabled:
                source = source_patches[i, j]
                target_y = np.int(np.clip(i + f[i, j, 0], 0, y - 1))
                target_x = np.int(np.clip(j + f[i, j, 1], 0, x - 1))
                target = target_patches[target_y, target_x]
                # distance of current patch of source img and it's corresponding target patch
                org_dis = get_distance(source, target)
                # Check if we are in odd iteration, then compare with left (i - 1, j) and up (i, j - 1)
                if odd_iteration:
                    left_org = None  # left patch of source img
                    up_org = None  # upper patch of source img
                    left_tar = None  # target patch corresponding to left patch of source img
                    up_tar = None  # target patch corresponding to upper patch of source img
                    left_dis = np.inf  # distance of left patch of source img and it's corresponding target patch
                    up_dis = np.inf  # distance of upper patch of source img and it's corresponding target patch
                    # when we are in first row, we only need care left
                    if (i - 1) < 0:
                        if (j - 1) >= 0:
                            # print i, j
                            up_org = source_patches[i, j - 1]
                    else:
                        # when we are in first col, we only need care up
                        if (j - 1) < 0:
                            # print i, j
                            left_org = source_patches[i - 1, j]
                        else:
                            up_org = source_patches[i, j - 1]
                            left_org = source_patches[i - 1, j]
                    # fill above info by case
                    if left_org is not None:
                        # It should be clamped by bound of img
                        left_tar_y = np.int(np.clip(i - 1 + new_f[i - 1, j, 0], 0, y - 1))
                        left_tar_x = np.int(np.clip(j + new_f[i - 1, j, 1], 0, x - 1))
                        left_tar = target_patches[left_tar_y, left_tar_x]
                        # print left_tar
                    if up_org is not None:
                        up_tar_y = np.int(np.clip(i + new_f[i, j - 1, 0], 0, y - 1))
                        up_tar_x = np.int(np.clip(j - 1 + new_f[i, j - 1, 1], 0, x - 1))
                        up_tar = target_patches[up_tar_y, up_tar_x]
                        # print up_tar
                    if left_tar is not None:
                        left_dis = get_distance(left_org, left_tar)
                    if up_tar is not None:
                        up_dis = get_distance(up_org, up_tar)
                    # update best_D and new_f
                    # print min([org_dis, left_dis, up_dis])
                    if min([org_dis, left_dis, up_dis]) == left_dis:
                        # print 'left'
                        new_f[i, j] = new_f[i - 1, j]
                        best_D[i, j] = left_dis
                    if min([org_dis, left_dis, up_dis]) == up_dis:
                        # print 'up'
                        new_f[i, j] = new_f[i, j - 1]
                        best_D[i, j] = up_dis
                    if min([org_dis, left_dis, up_dis]) == org_dis:
                        best_D[i, j] = org_dis
                # even loop (revers order)
                else:
                    right_org = None  # right patch of source img
                    down_org = None  # lower patch of source img
                    right_tar = None  # target patch corresponding to right patch of source img
                    down_tar = None  # target patch corresponding to lower patch of source img
                    right_dis = np.inf  # distance of right patch of source img and it's corresponding target patch
                    down_dis = np.inf  # distance of lower patch of source img and it's corresponding target patch
                    # when we are in last row, we only need care right
                    if i + 1 > y - 1:
                        if (j + 1) <= x - 1:
                            down_org = source_patches[i, j + 1]
                    else:
                        # when we are in last col, we only need care down
                        if j + 1 > x - 1:
                            right_org = source_patches[i + 1, j]
                        else:
                            down_org = source_patches[i, j + 1]
                            right_org = source_patches[i + 1, j]
                    # fill above info by case
                    if right_org is not None:
                        # It should be clamped by bound of img
                        right_tar_y = np.int(np.clip(i + 1 + new_f[i + 1, j, 0], 0, y - 1))
                        right_tar_x = np.int(np.clip(j + new_f[i + 1, j, 1], 0, x - 1))
                        right_tar = target_patches[right_tar_y, right_tar_x]
                    if down_org is not None:
                        down_tar_y = np.int(np.clip(i + new_f[i, j + 1, 0], 0, y - 1))
                        down_tar_x = np.int(np.clip(j + 1 + new_f[i, j + 1, 1], 0, x - 1))
                        down_tar = target_patches[down_tar_y, down_tar_x]
                    if right_tar is not None:
                        right_dis = get_distance(right_org, right_tar)
                    if down_tar is not None:
                        down_dis = get_distance(down_org, down_tar)
                    # update best_D and new_f
                    # print min([org_dis, right_dis, down_dis])
                    if min([org_dis, right_dis, down_dis]) == right_dis:
                        # print 'right'
                        new_f[i, j] = new_f[i + 1, j]
                        best_D[i, j] = right_dis
                    if min([org_dis, right_dis, down_dis]) == down_dis:
                        # print 'down'
                        new_f[i, j] = new_f[i, j + 1]
                        best_D[i, j] = down_dis
                    if min([org_dis, right_dis, down_dis]) == org_dis:
                        best_D[i, j] = org_dis
            # See if we want to preform random search
            if random_enabled:
                k = 0
                while alpha ** k * w >= 1:
                    R = np.random.uniform(-1, 1, 2)
                    u_k = new_f[i, j] + (alpha ** k) * w * R
                    # It should be clamped by bound of B
                    random_target_y = np.int(np.clip(i + u_k[0], 0, y - 1))
                    random_target_x = np.int(np.clip(j + u_k[1], 0, x - 1))
                    random_target = target_patches[random_target_y, random_target_x]
                    # find out the distance
                    temp_dist = get_distance(random_target, source_patches[i, j])
                    # print temp_dist
                    k += 1
                    # print u_k, temp_dist
                    # print random_target_y, random_target_x
                    # print random_target, source_patches[i, j]
                    # update best_D and new_f
                    if temp_dist < best_D[i, j]:
                        best_D[i, j] = temp_dist
                        new_f[i, j] = u_k

    #############################################

    return global_vars


# This function builds a 2D heap data structure to represent the k nearest-neighbour
# fields supplied as input to the function.
#
# The function takes three input arguments:
#     - source_patches:      The matrix holding the patches of the source image (see above)
#     - target_patches:      The matrix holding the patches of the target image (see above)
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds k NNFs. Specifically,
#                            f_k[i] is the i-th NNF and has dimension NxMx2 for an NxM image.
#                            There is NO requirement that f_k[i] corresponds to the i-th best NNF,
#                            i.e., f_k is simply assumed to be a matrix of vector fields.
#
# The function should return the following two data structures:
#     - f_heap:              A 2D array of heaps. For an NxM image, this array is represented as follows:
#                               * f_heap is a list of length N, one per image row
#                               * f_heap[i] is a list of length M, one per pixel in row i
#                               * f_heap[i][j] is the heap of pixel (i,j)
#                            The heap f_heap[i][j] should contain exactly k tuples, one for each
#                            of the 2D displacements f_k[0][i][j],...,f_k[k-1][i][j]
#
#                            Each tuple has the format: (priority, counter, displacement)
#                            where
#                                * priority is the value according to which the tuple will be ordered
#                                  in the heapq data structure
#                                * displacement is equal to one of the 2D vectors
#                                  f_k[0][i][j],...,f_k[k-1][i][j]
#                                * counter is a unique integer that is assigned to each tuple for
#                                  tie-breaking purposes (ie. in case there are two tuples with
#                                  identical priority in the heap)
#     - f_coord_dictionary:  A 2D array of dictionaries, represented as a list of lists of dictionaries.
#                            Specifically, f_coord_dictionary[i][j] should contain a dictionary
#                            entry for each displacement vector (x,y) contained in the heap f_heap[i][j]
#
# NOTE: This function should NOT check for duplicate entries or out-of-bounds vectors
# in the heap: it is assumed that the heap returned by this function contains EXACTLY k tuples
# per pixel, some of which MAY be duplicates or may point outside the image borders

def NNF_matrix_to_NNF_heap(source_patches, target_patches, f_k):

    f_heap = None
    f_coord_dictionary = None

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    # initialize f_heap and f_coord_dictionary
    f_heap = []
    f_coord_dictionary = []
    y, x = source_patches.shape[:2]
    k = f_k.shape[0]
    for i in range(y):
        # for each row.
        heap = []
        coord = []
        for j in range(x):
            heap_temp = []
            dic_temp = {}
            for l in range(k):
                offset = f_k[l, i, j]
                # check if out of bound
                tar_y = np.clip(i + offset[0], 0, y - 1)
                tar_x = np.clip(j + offset[1], 0, x - 1)
                tar_patch = target_patches[tar_y, tar_x]
                # use the distance as the priority
                pro = -get_distance(source_patches[i, j], tar_patch)
                heappush(heap_temp, (pro, _tiebreaker.next(), offset))
                dic_temp[tuple(offset)] = 0
            heap.append(heap_temp)
            coord.append(dic_temp)
        f_heap.append(heap)
        f_coord_dictionary.append(coord)

    #############################################

    return f_heap, f_coord_dictionary


# Given a 2D array of heaps given as input, this function creates a kxNxMx2
# matrix of nearest-neighbour fields
#
# The function takes only one input argument:
#     - f_heap:              A 2D array of heaps as described above. It is assumed that
#                            the heap of every pixel has exactly k elements.
# and has two return arguments
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds the k NNFs represented by the heap.
#                            Specifically, f_k[i] should be the NNF that contains the i-th best
#                            displacement vector for all pixels. Ie. f_k[0] is the best NNF,
#                            f_k[1] is the 2nd-best NNF, f_k[2] is the 3rd-best, etc.
#     - D_k:                 A numpy array of dimensions kxNxM whose element D_k[i][r][c] is the patch distance
#                            corresponding to the displacement f_k[i][r][c]
#

def NNF_heap_to_NNF_matrix(f_heap):

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    # print f_heap
    y = len(f_heap)
    x = len(f_heap[0])
    k = len(f_heap[0, 0])
    # print nlargest(k, f_heap[0, 0])
    # initialize f_k and D_k
    f_k = np.zeros((k, y, x, 2))
    D_k = np.zeros((k, y, x))
    for i in range(y):
        for j in range(x):
            for l in range(k):
                # fill f_k and D_k
                # here make sure the displacements are filled in order.
                f_k[l, i, j] = nlargest(l + 1, f_heap[i, j])[l, 2, :]
                # the priority is the distance
                D_k[l, i, j] = -(nlargest(l + 1, f_heap[i, j])[l, 0, :])
    #############################################

    return f_k, D_k


def nlm(target, f_heap, h):


    # this is a dummy statement to return the image given as input
    # denoised = target

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    # initialize denoised image
    denoised = np.zeros(target.shape)
    y, x = target.shape[:2]

    #############################################

    return denoised


#############################################
###  PLACE ADDITIONAL HELPER ROUTINES, IF ###
###  ANY, BETWEEN THESE LINES             ###
#############################################
# Helper for computing D

def get_distance(source, target):
    # print source.shape, target.shape
    dist = np.subtract(source, target)
    return np.linalg.norm(np.abs(dist))


#############################################


# This function uses a computed NNF to reconstruct the source image
# using pixels from the target image. The function takes two input
# arguments
#     - target: the target image that was used as input to PatchMatch
#     - f:      the nearest-neighbor field the algorithm computed
# and should return a reconstruction of the source image:
#     - rec_source: an openCV image that has the same shape as the source image
#
# To reconstruct the source, the function copies to pixel (x,y) of the source
# the color of pixel (x,y)+f(x,y) of the target.
#
# The goal of this routine is to demonstrate the quality of the computed NNF f.
# Specifically, if patch (x,y)+f(x,y) in the target image is indeed very similar
# to patch (x,y) in the source, then copying the color of target pixel (x,y)+f(x,y)
# to the source pixel (x,y) should not change the source image appreciably.
# If the NNF is not very high quality, however, the reconstruction of source image
# will not be very good.
#
# You should use matrix/vector operations to avoid looping over pixels,
# as this would be very inefficient

def reconstruct_source_from_target(target, f):
    rec_source = None

    ################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES  ###
    ################################################
    y, x = target.shape[:2]
    # clamp
    target_coor = make_coordinates_matrix([y, x]) + f
    tar_idx_y = np.clip(target_coor[:, :, 0], 0, y - 1)
    tar_idx_x = np.clip(target_coor[:, :, 1], 0, x - 1)
    rec_source = target[tar_idx_y, tar_idx_x]

    #############################################

    return rec_source


# This function takes an NxM image with C color channels and a patch size P
# and returns a matrix of size NxMxCxP^2 that contains, for each pixel [i,j] in
# in the image, the pixels in the patch centered at [i,j].
#
# You should study this function very carefully to understand precisely
# how pixel data are organized, and how patches that extend beyond
# the image border are handled.


def make_patch_matrix(im, patch_size):
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


# Generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
# such that g(y,x) = [y,x]
#
# Step is an optional argument used to create a matrix that is step times
# smaller than the full image in each dimension
#
# Pay attention to this function as it shows how to perform these types
# of operations in a vectorized manner, without resorting to loops


def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))
