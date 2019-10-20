# CSC320 Winter 2019
# Assignment 3
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

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the PatchMatch
# algorithm, as explained in Section 3.2 of the paper.
# The function takes an NNF f as input, performs propagation and random search,
# and returns an updated NNF.
#
# The function takes several input arguments:
#     - source_patches:      The matrix holding the patches of the source image,
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
#     - target_patches:      The matrix holding the patches of the target image.
#     - f:                   The current nearest-neighbour field
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
#     - best_D:              And NxM matrix whose element [i,j] is the similarity score between
#                            patch [i,j] in the source and its best-matching patch in the
#                            target. Use this matrix to check if you have found a better
#                            match to [i,j] in the current PatchMatch iteration
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - new_f:               The updated NNF
#     - best_D:              The updated similarity scores for the best-matching patches in the
#                            target
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure


def propagation_and_random_search(source_patches, target_patches,
                                  f, alpha, w,
                                  propagation_enabled, random_enabled,
                                  odd_iteration, best_D=None,
                                  global_vars=None
                                ):
    new_f = f.copy()

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    # # print odd_iteration
    # y, x = source_patches.shape[:2]
    # # (285, 630, 3, 49) (285, 630, 3, 49) (285, 630, 2)
    # # print source_patches.shape, target_patches.shape, f.shape
    # # initial our D matrix (should be large at first)
    # if best_D is None:
    #     best_D = np.multiply(np.ones((y, x)), np.inf)
    # # set all nan value to 0
    # # source_patches = np.nan_to_num(source_patches)
    # # target_patches = np.nan_to_num(target_patches)
    #
    # # Firstly see which iteration we are at, and decide how do we loop.
    # if odd_iteration:
    #     # print "odd"
    #     loop_i = range(y)
    #     loop_j = range(x)
    #     dir = -1 # according to the paper
    #     bond_i, bond_j = 0, 0
    # else:
    #     loop_i = range(y - 1, -1, -1)
    #     loop_j = range(x - 1, -1, -1)
    #     dir = 1
    #     bond_i, bond_j = y - 1, x - 1 # science we scan it in revers direction
    #     # See if we want to preform propagation
    # for i in loop_i:
    #     for j in loop_j:
    #         if propagation_enabled:
    #             # print 'prop'
    #             # source = source_patches[i, j]
    #             # target_y = np.int(np.clip(i + f[i, j, 0], 0, y - 1))
    #             # target_x = np.int(np.clip(j + f[i, j, 1], 0, x - 1))
    #             # target = target_patches[target_y, target_x]
    #             # # distance of current patch of source img and it's corresponding target patch
    #             # org_dis = get_distance(source, target)
    #             # # Check if we are in odd iteration, then compare with left (i - 1, j) and up (i, j - 1)
    #             # if odd_iteration:
    #                 # print 'odd'
    #             # nib1_org = None # a neighbour patch of source img
    #             # nib2_org = None # a neighbour patch of source img
    #             # nib1_tar = None # target patch corresponding to neighbour patch of source img
    #             # nib2_tar = None # target patch corresponding to neighbour patch of source img
    #             # nib1_dis = np.inf # distance of neighbour patch of source img and it's corresponding target patch
    #             # nib2_dis = np.inf # distance of neighbour patch of source img and it's corresponding target patch
    #             # # we don't need to care the first patch
    #             org_patch = [np.array([i, j])] # this list contains the patch idx that we need to compare
    #             coor_f = [new_f[i, j]] # This stores the corresponding f of the patch
    #
    #             if i == bond_i:
    #                 if j != bond_j:
    #                     org_patch.append([i, j + dir])
    #                     coor_f.append(f[i, j + dir])
    #             else:
    #                 # while i not reach bond
    #                 if j == bond_j:
    #                     org_patch.append([i + dir, j])
    #                     coor_f.append(f[i + dir, j])
    #                 else:
    #                     org_patch.append([i + dir, j])
    #                     coor_f.append(f[i + dir, j])
    #                     org_patch.append([i, j + dir])
    #                     coor_f.append(f[i, j + dir])
    #             # # fill above info by case
    #             # if nib1_org is not None:
    #             #     # It should be clamped by bound of img
    #             #     # nib1_tar_y = np.int(np.clip(i + dir + f[i + dir, j, 0], 0, y - 1))
    #             #     # nib1_tar_x = np.int(np.clip(j + f[i + dir, j, 1], 0, x - 1))
    #             #     tar_idx = np.array([i, j + dir]) + f[i, j + dir]
    #             #     tar_idx_y = np.clip(tar_idx[0], 0, y - 1)
    #             #     tar_idx_x = np.clip(tar_idx[1], 0, x - 1)
    #             #     nib1_tar = target_patches[tar_idx_y, tar_idx_x]
    #             # if nib2_org is not None:
    #             #     # nib2_tar_y = np.int(np.clip(i + f[i, j + dir, 0], 0, y - 1))
    #             #     # nib2_tar_x = np.int(np.clip(j + dir + f[i, j + dir, 1], 0, x - 1))
    #             #     tar_idx = np.array([i + dir, j]) + f[i + dir, j]
    #             #     tar_idx_y = np.clip(tar_idx[0], 0, y - 1)
    #             #     tar_idx_x = np.clip(tar_idx[1], 0, x - 1)
    #             #     nib2_tar = target_patches[tar_idx_y, tar_idx_x]
    #             # if nib1_tar is not None:
    #             #     nib1_dis = get_distance(nib1_org, nib1_tar)
    #             # if nib2_tar is not None:
    #             #     nib2_dis = get_distance(nib2_org, nib2_tar)
    #             # # update best_D and new_f
    #             # if min([org_dis, nib1_dis, nib2_dis]) == nib1_dis:
    #             #     new_f[i, j] = f[i + dir, j]
    #             #     best_D[i, j] = nib1_dis
    #             # if min([org_dis, nib1_dis, nib2_dis]) == nib2_dis:
    #             #     new_f[i, j] = f[i, j + dir]
    #             #     best_D[i, j] = nib2_dis
    #             for k in range(len(org_patch)):
    #                 tar_idx = org_patch[k] + coor_f[k]
    #                 # clamp
    #                 tar_idx_y = np.clip(tar_idx[0], 0, y - 1)
    #                 tar_idx_x = np.clip(tar_idx[1], 0, x - 1)
    #                 tar = target_patches[tar_idx_y, tar_idx_x]
    #                 dist = get_distance(source_patches[coor_f[k]], tar)
    #                 # update
    #                 if dist < best_D[i, j]:
    #                     new_f[i, j] = coor_f[k]
    #                     best_D[i, j] = dist
    #         # even loop (revers order)
    #             # else:
    #             #     # print 'even'
    #             #     right_org = None  # right patch of source img
    #             #     down_org = None  # lower patch of source img
    #             #     right_tar = None  # target patch corresponding to right patch of source img
    #             #     down_tar = None  # target patch corresponding to lower patch of source img
    #             #     right_dis = np.inf  # distance of right patch of source img and it's corresponding target patch
    #             #     down_dis = np.inf  # distance of lower patch of source img and it's corresponding target patch
    #             #     # when we are in last row, we only need care right
    #             #     if i + 1 > y - 1:
    #             #         if (j + 1) < x - 1:
    #             #             down_org = source_patches[i, j + 1]
    #             #     else:
    #             #         # when we are in last col, we only need care down
    #             #         if j + 1 > x - 1:
    #             #             down_org = source_patches[i + 1, j]
    #             #         else:
    #             #             down_org = source_patches[i, j + 1]
    #             #             right_org = source_patches[i + 1, j]
    #             #     # fill above info by case
    #             #     if right_org is not None:
    #             #         # It should be clamped by bound of img
    #             #         right_tar_y = np.int(np.clip(i + 1 + f[i + 1, j, 0], 0, y - 1))
    #             #         right_tar_x = np.int(np.clip(j + f[i + 1, j, 1], 0, x - 1))
    #             #         right_tar = target_patches[right_tar_y, right_tar_x]
    #             #     if down_org is not None:
    #             #         down_tar_y = np.int(np.clip(i + f[i, j + 1, 0], 0, y - 1))
    #             #         down_tar_x = np.int(np.clip(j + 1 + f[i, j + 1, 1], 0, x - 1))
    #             #         down_tar = target_patches[down_tar_y, down_tar_x]
    #             #     if right_tar is not None:
    #             #         right_dis = get_distance(right_org, right_tar)
    #             #     if down_tar is not None:
    #             #         down_dis = get_distance(down_org, down_tar)
    #             #     # update best_D and new_f
    #             #     if min([org_dis, right_dis, down_dis]) == right_dis:
    #             #         new_f[i, j] = f[i + 1, j]
    #             #         best_D[i, j] = right_dis
    #             #     if min([org_dis, right_dis, down_dis]) == down_dis:
    #             #         new_f[i, j] = f[i, j + 1]
    #             #         best_D[i, j] = down_dis
    #             # print "prop"
    #         # See if we want to preform random search
    #         if random_enabled:
    #             k = 0
    #             while (alpha ** k) * w >= 1:
    #                 R = np.random.uniform(-1, 1, 2)
    #                 u_k = f[i, j] + (alpha ** k) * w * R
    #                 # It should be clamped by bound of B
    #                 random_target_y = np.int(i + u_k[0])
    #                 random_target_x = np.int(j + u_k[1])
    #                 if random_target_x < 0:
    #                     random_target_x = 0
    #                     u_k[1] = - j
    #                 if random_target_x > x - 1:
    #                     random_target_x = x - 1
    #                     u_k[1] = x - 1 - j
    #                 if random_target_y < 0:
    #                     random_target_y = 0
    #                     u_k[0] = - i
    #                 if random_target_y > y - 1:
    #                     random_target_y = y - 1
    #                     u_k[0] = y - 1 - i
    #                 random_target = target_patches[random_target_y, random_target_x]
    #                 # find out the distance
    #                 temp_dist = get_distance(random_target, source_patches[i, j])
    #                 # print temp_dist
    #                 k += 1
    #                 # print u_k, temp_dist
    #                 # print random_target_y, random_target_x
    #                 # print random_target, source_patches[i, j]
    #                 # update best_D and new_f
    #                 if temp_dist < best_D[i, j]:
    #                     best_D[i, j] = temp_dist
    #                     new_f[i, j] = u_k
    #         # if random_enabled:
    #         #     k = 0
    #         #     while (alpha ** k) * w >= 1:
    #         #         R = np.random.uniform(-1, 1, 2)
    #         #         u_k = f[i, j] + (alpha ** k) * w * R
    #         #         # It should be clamped by bound of B
    #         #         random_target_y = np.int(np.clip(i + u_k[0], 0, y - 1))
    #         #         random_target_x = np.int(np.clip(j + u_k[1], 0, x - 1))
    #         #         random_target = target_patches[random_target_y, random_target_x]
    #         #         # find out the distance
    #         #         temp_dist = get_distance(random_target, source_patches[i, j])
    #         #         # print temp_dist
    #         #         k += 1
    #         #         # print u_k, temp_dist
    #         #         # print random_target_y, random_target_x
    #         #         # print random_target, source_patches[i, j]
    #         #         # update best_D and new_f
    #         #         if temp_dist < best_D[i, j]:
    #         #             best_D[i, j] = temp_dist
    #         #             new_f[i, j] = u_k
    y, x = source_patches.shape[:2]
    # (285, 630, 3, 49) (285, 630, 3, 49) (285, 630, 2)
    # print source_patches.shape, target_patches.shape, f.shape
    # initial our D matrix (should be large at first)
    if best_D is None:
        best_D = np.multiply(np.ones((y, x)), np.inf)
    # replace NaN with zero in both patch
    source_patches = np.nan_to_num(source_patches)
    target_patches = np.nan_to_num(target_patches)
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
                # source = source_patches[i, j]
                # target_y = np.int(np.clip(i + f[i, j, 0], 0, y - 1))
                # target_x = np.int(np.clip(j + f[i, j, 1], 0, x - 1))
                # target = target_patches[target_y, target_x]
                # distance of current patch of source img and it's corresponding target patch
                # org_dis = get_distance(source, target)
                org_dis = best_D[i, j]
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
                        if (j - 1) > 0:
                            up_org = source_patches[i, j - 1]
                    else:
                        # when we are in first col, we only need care up
                        if (j - 1) < 0:
                            left_org = source_patches[i - 1, j]
                        else:
                            up_org = source_patches[i, j - 1]
                            left_org = source_patches[i - 1, j]
                    # fill above info by case
                    if left_org is not None:
                        # It should be clamped by bound of img
                        left_tar_y = np.int(np.clip(i - 1 + f[i - 1, j, 0], 0, y - 1))
                        left_tar_x = np.int(np.clip(j + f[i - 1, j, 1], 0, x - 1))
                        left_tar = target_patches[left_tar_y, left_tar_x]
                    if up_org is not None:
                        up_tar_y = np.int(np.clip(i + f[i, j - 1, 0], 0, y - 1))
                        up_tar_x = np.int(np.clip(j - 1 + f[i, j - 1, 1], 0, x - 1))
                        up_tar = target_patches[up_tar_y, up_tar_x]
                    if left_tar is not None:
                        left_dis = get_distance(left_org, left_tar)
                    if up_tar is not None:
                        up_dis = get_distance(up_org, up_tar)
                    # update best_D and new_f
                    if min([org_dis, left_dis, up_dis]) == left_dis:
                        new_f[i, j] = f[i - 1, j]
                        best_D[i, j] = left_dis
                    if min([org_dis, left_dis, up_dis]) == up_dis:
                        new_f[i, j] = f[i, j - 1]
                        best_D[i, j] = up_dis
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
                        if (j + 1) < x - 1:
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
                        right_tar_y = np.int(np.clip(i + 1 + f[i + 1, j, 0], 0, y - 1))
                        right_tar_x = np.int(np.clip(j + f[i + 1, j, 1], 0, x - 1))
                        right_tar = target_patches[right_tar_y, right_tar_x]
                    if down_org is not None:
                        down_tar_y = np.int(np.clip(i + f[i, j + 1, 0], 0, y - 1))
                        down_tar_x = np.int(np.clip(j + 1 + f[i, j + 1, 1], 0, x - 1))
                        down_tar = target_patches[down_tar_y, down_tar_x]
                    if right_tar is not None:
                        right_dis = get_distance(right_org, right_tar)
                    if down_tar is not None:
                        down_dis = get_distance(down_org, down_tar)
                    # update best_D and new_f
                    if min([org_dis, right_dis, down_dis]) == right_dis:
                        new_f[i, j] = f[i + 1, j]
                        best_D[i, j] = right_dis
                    if min([org_dis, right_dis, down_dis]) == down_dis:
                        new_f[i, j] = f[i, j + 1]
                        best_D[i, j] = right_dis
            # See if we want to preform random search
            if random_enabled:
                k = 0
                while alpha ** k * w >= 1:
                    R = np.random.uniform(-1, 1, 2)
                    u_k = f[i, j] + (alpha ** k) * w * R
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

    return new_f, best_D, global_vars

# Helper for computing D


def get_distance(source, target):
    # print source.shape, target.shape
    dist = np.subtract(source, target)
    return np.linalg.norm(np.abs(dist))


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

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    y, x = target.shape[:2]
    # clamp
    target_coor = make_coordinates_matrix([y, x]) + f
    tar_idx_y = np.clip(target_coor[:, :, 0], 0, y - 1)
    tar_idx_x = np.clip(target_coor[:, :, 1], 0, x - 1)
    rec_source = target[tar_idx_y, tar_idx_x]

    # ((285, 630, 2), (285, 630, 3), (285, 630, 2))
    # print f.shape, target.shape, target_coor.shape

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
