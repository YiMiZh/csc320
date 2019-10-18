import numpy as np
import scipy.linalg as sp
import cv2 as cv


# a = np.array([[[1,2,3]],[[1,2,3]]])
# b = np.array([[[4,5,6]],[[1,2,3]]])
# # print a, "\n", b
# # print(b - a)
# # print np.identity(3)
# a = np.array([[[1, 2], [3, 4]],[[11, 22], [33, 44]]])
# b = np.array([[[111, 222], [333, 444]],[[1111, 2222], [3333, 4444]]])
# # print a, "\n", b, "\n", np.concatenate((a, b), axis=2)
# matrix = np.tile(np.identity(3), (2, 1))
# many_matrices = np.tile(matrix, (480, 320, 1, 1))
# E = np.tile(np.eye(3), (480,320,1,1))
# aa = np.multiply(backA.reshape(M,N,3,1), -1)
# ab = np.multiply(backB.reshape(M,N,3,1), -1)
# aa = np.concatenate((E, aa), axis=3)
# ab = np.concatenate((E, ab), axis=3)
# A = np.concatenate((aa,ab), axis=2)
a = np.array([[[1,2,3],
               [2,3,4]],
              [[1,2,3],
               [4,5,6]]])
E = np.tile(np.eye(3), (2,2,1,1))
aa = a.reshape(2,2,3, 1)
aaa = np.concatenate((E, aa), axis=3)
A = np.concatenate((aaa,aaa), axis=2)
print E, E.shape, aa, aa.shape, aaa, aaa.shape
print A, A.shape