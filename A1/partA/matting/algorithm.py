## CSC320 Winter 2019 
## Assignment 1
## (c) Kyros Kutulakos
##
## DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
## AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION 
## BY THE INSTRUCTOR IS STRICTLY PROHIBITED. VIOLATION OF THIS 
## POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

##
## DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
##

# import basic packages
import numpy as np
import scipy.linalg as sp
import cv2 as cv


# If you wish to import any additional modules
# or define other utility functions, 
# include them here

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################


#########################################

#
# The Matting Class
#
# This class contains all methods required for implementing 
# triangulation matting and image compositing. Description of
# the individual methods is given below.
#
# To run triangulation matting you must create an instance
# of this class. See function run() in file run.py for an
# example of how it is called
#
class Matting:
    #
    # The class constructor
    #
    # When called, it creates a private dictionary object that acts as a container
    # for all input and all output images of the triangulation matting and compositing 
    # algorithms. These images are initialized to None and populated/accessed by 
    # calling the the readImage(), writeImage(), useTriangulationResults() methods.
    # See function run() in run.py for examples of their usage.
    #
    def __init__(self):
        self._images = {
            'backA': None,
            'backB': None,
            'compA': None,
            'compB': None,
            'colOut': None,
            'alphaOut': None,
            'backIn': None,
            'colIn': None,
            'alphaIn': None,
            'compOut': None,
        }

    # Return a dictionary containing the input arguments of the
    # triangulation matting algorithm, along with a brief explanation
    # and a default filename (or None)
    # This dictionary is used to create the command-line arguments
    # required by the algorithm. See the parseArguments() function
    # run.py for examples of its usage
    def mattingInput(self):
        return {
            'backA': {'msg': 'Image filename for Background A Color', 'default': None},
            'backB': {'msg': 'Image filename for Background B Color', 'default': None},
            'compA': {'msg': 'Image filename for Composite A Color', 'default': None},
            'compB': {'msg': 'Image filename for Composite B Color', 'default': None},
        }

    # Same as above, but for the output arguments
    def mattingOutput(self):
        return {
            'colOut': {'msg': 'Image filename for Object Color', 'default': ['color.tif']},
            'alphaOut': {'msg': 'Image filename for Object Alpha', 'default': ['alpha.tif']}
        }

    def compositingInput(self):
        return {
            'colIn': {'msg': 'Image filename for Object Color', 'default': None},
            'alphaIn': {'msg': 'Image filename for Object Alpha', 'default': None},
            'backIn': {'msg': 'Image filename for Background Color', 'default': None},
        }

    def compositingOutput(self):
        return {
            'compOut': {'msg': 'Image filename for Composite Color', 'default': ['comp.tif']},
        }

    # Copy the output of the triangulation matting algorithm (i.e., the 
    # object Color and object Alpha images) to the images holding the input
    # to the compositing algorithm. This way we can do compositing right after
    # triangulation matting without having to save the object Color and object
    # Alpha images to disk. This routine is NOT used for partA of the assignment.
    def useTriangulationResults(self):
        if (self._images['colOut'] is not None) and (self._images['alphaOut'] is not None):
            self._images['colIn'] = self._images['colOut'].copy()
            self._images['alphaIn'] = self._images['alphaOut'].copy()

    # If you wish to create additional methods for the 
    # Matting class, include them here

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    #########################################

    #########################################

    # Use OpenCV to read an image from a file and copy its contents to the 
    # matting instance's private dictionary object. The key 
    # specifies the image variable and should be one of the
    # strings in lines 54-63. See run() in run.py for examples
    #
    # The routine should return True if it succeeded. If it did not, it should
    # leave the matting instance's dictionary entry unaffected and return
    # False, along with an error message
    def readImage(self, fileName, key):
        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        image = cv.imread(fileName)
        # print(image)
        # cv.imshow("hi", image)
        # cv.waitKey(2000)
        if image is None:
            msg = 'Read failed'
        else:
            self._images[key] = image
            success = True
        #########################################
        return success, msg

    # Use OpenCV to write to a file an image that is contained in the 
    # instance's private dictionary. The key specifies the which image
    # should be written and should be one of the strings in lines 54-63. 
    # See run() in run.py for usage examples
    #
    # The routine should return True if it succeeded. If it did not, it should
    # return False, along with an error message
    def writeImage(self, fileName, key):
        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        if cv.imwrite(fileName, self._images[key]):
            success = True
        else:
            msg = 'Write failed'
        #########################################
        return success, msg

    # Method implementing the triangulation matting algorithm. The
    # method takes its inputs/outputs from the method's private dictionary 
    # ojbect. 
    def triangulationMatting(self):
        """
        success, errorMessage = triangulationMatting(self)
        
        Perform triangulation matting. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
        """

        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        if (self._images['compA'] is None) or \
                (self._images['compB'] is None) or \
                (self._images['backA'] is None) or \
                (self._images['backB'] is None):
            msg = "Empty image"
            return success, msg

        compA = np.divide(self._images['compA'], 255.)
        compB = np.divide(self._images['compB'], 255.)
        backA = np.divide(self._images['backA'], 255.)
        backB = np.divide(self._images['backB'], 255.)
        # print self._images['backA'].shape
        row, col, rgb = self._images['backA'].shape
        colOut = np.zeros((row, col, rgb))
        alphaOut = np.zeros((row, col))
        # print row, col, rgb
        idmat = np.identity(3)
        id_mat = np.concatenate((idmat, idmat))
        # print id_mat
        delta_cA = compA - backA
        delta_cB = compB - backB
        # print self._images['compA'][:, :, 0], self._images['backA'][:, :, 0], delta_cA[:, :, 0]
        # print delta_cA[:, :, 0]
        # print delta_cA.shape
        # print np.concatenate((backA[0][0], backB[0][0])).reshape(6, 1)
        for i in range(row):
            for j in range(col):
                temp = np.concatenate((backA[i, j], backB[i, j])).reshape(6, 1)
                A = np.concatenate((id_mat, -temp), axis=1)
                # print A.shape
                delta_C = np.concatenate((delta_cA[i, j], delta_cB[i, j]))
                # print delta_C.shape
                # x = np.dot(np.linalg.pinv(A), delta_C)
                x = np.linalg.lstsq(A, delta_C, rcond=None)[0]
                # print x
                colOut[i, j] = np.clip(x[0:3], 0., 1.)
                alphaOut[i, j] = np.clip(x[3], 0., 1.)
            #     break
            # break
        self._images['colOut'] = colOut * 255
        self._images['alphaOut'] = alphaOut * 255
        # print self._images['colOut'], "\n", self._images['alphaOut']
        success = True
        #########################################

        return success, msg

    def createComposite(self):
        """
success, errorMessage = createComposite(self)
        
        Perform compositing. Returns True if successful (ie.
        all inputs and outputs are valid) and False if not. When success=False
        an explanatory error message should be returned.
"""

        success = False
        msg = 'Placeholder'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        if (self._images['backIn'] is None) or \
                (self._images['alphaIn'] is None) or \
                (self._images['colIn'] is None):
            msg = "Wrong input"
            return success, msg
        backIn = np.divide(self._images['backIn'], 255.)
        alphaIn = np.divide(self._images['alphaIn'], 255.)
        colIn = np.divide(self._images['colIn'], 255.)
        compOut = colIn + (1. - alphaIn) * backIn
        self._images['compOut'] = compOut * 255.
        success = True
        #########################################

        return success, msg
