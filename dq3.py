########### Assumption #############:
# 1. BackGround color is of same color throughout,
#    with no objects or noise present.
#
# 2. No Wearing of skin color cloth
####################################

##### Alternate:
# 1. Allow wearing of skin color cloth, detecting face to remove face skin color
#    and also detect the leg and hands; and same then remove skin color.
#
# 2. Removal of Hair from the image to get good info
#
#####

import numpy as np
import argparse
import cv2


# BackGround Substraction
def background_sub(img, BLUR = 21, CANNY_THRESH_1 = 10, CANNY_THRESH_2 = 70, MASK_DILATE_ITER = 10, MASK_ERODE_ITER = 10, MASK_COLOR = (1.0,1.0,1.0)):
	
	
	#== Processing =======================================================================
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	#-- Edge detection -------------------------------------------------------------------
	edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
	edges = cv2.dilate(edges, None)
	edges = cv2.erode(edges, None)

	#-- Find contours in edges, sort by area ---------------------------------------------
	contour_info = []
	_, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	for c in contours:
	    contour_info.append((
		c,
		cv2.isContourConvex(c),
		cv2.contourArea(c),
	    ))
	contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
	max_contour = contour_info[0]

	#-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
	# Mask is black, polygon is white
	mask = np.zeros(edges.shape)
	cv2.fillConvexPoly(mask, max_contour[0], (255))

	#-- Smooth mask, then blur it --------------------------------------------------------
	mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
	mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
	mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
	mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

	#-- Blend masked img into MASK_COLOR background --------------------------------------
	mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
	img         = img.astype('float32') / 255.0                 #  for easy blending

	masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
	masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 

	return masked


# Skin Color Removal
def skin_color_deletion(masked, lower, upper):
	#RGB
	converted = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
	skinMask = cv2.inRange(converted, lower, upper)

	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
	skinMask = cv2.erode(skinMask, kernel, iterations = 7)
	skinMask = cv2.dilate(skinMask, kernel, iterations = 7)

	# blur the mask to help remove noise, then apply the
	# mask to the frame
	skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
	skin = cv2.bitwise_and(masked, masked, mask = ~skinMask)
	inds = skin[:,:,1] < 1
	skin[inds] = [255, 255, 255]
	return skin



'''
#== Parameters =======================================================================
	BLUR = 21
	CANNY_THRESH_1 = 10
	CANNY_THRESH_2 = 70
	MASK_DILATE_ITER = 10
	MASK_ERODE_ITER = 10
	MASK_COLOR = (1.0,1.0,1.0) # In BGR format
'''


# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-b", "--blur", required=False, type = int, default = 9,
	help="serial no to input image")
args = vars(ap.parse_args())


#-- Read image -----------------------------------------------------------------------
img = cv2.imread(args["image"])
BLUR = args["blur"]


# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")


no_bkgd = background_sub(img)
no_skin = skin_color_deletion(no_bkgd, lower, upper)

if(args["blur"]):
	no_skin = cv2.GaussianBlur(no_skin, (BLUR, BLUR), 0)

# Masked HSV for Red, Green, Blue Color
boundaries = [
		([164, 80, 60], [184, 250, 250]),
		([29, 83, 47], [66, 255, 255]),
		([100, 13, 14], [129, 255, 255])
]

boundaries = np.asarray(boundaries)
converted = cv2.cvtColor(no_skin, cv2.COLOR_BGR2HSV)

# Masking the No-Skin Image with Red, Green, Blue Color Mask
total_pixel = converted.shape[0] * converted.shape[1]
white_pixel = np.count_nonzero((no_skin == [255, 255, 255]).all(axis = 2))
total_pixel = total_pixel - white_pixel
total_pixel = float(total_pixel)


redMask = cv2.inRange(converted, boundaries[0][0], boundaries[0][1])
red = cv2.bitwise_and(converted, converted, mask = redMask)
red_pixel = np.count_nonzero((red > [0, 0, 0]).all(axis = 2))

greenMask = cv2.inRange(converted, boundaries[1][0], boundaries[1][1])
green = cv2.bitwise_and(converted, converted, mask = greenMask)
green_pixel = np.count_nonzero((green > [0, 0, 0]).all(axis = 2))

blueMask = cv2.inRange(converted, boundaries[2][0], boundaries[2][1])
blue = cv2.bitwise_and(converted, converted, mask = blueMask)
blue_pixel = np.count_nonzero((blue > [0, 0, 0]).all(axis = 2))

# Counting the Pixel Count of BGR
tot = 100./total_pixel
print "Red:  ", red_pixel*tot, "%\n",  "Green:", green_pixel*tot, "%\n", "Blue: ", blue_pixel*tot, "%\n"



# Showing of an Image
cv2.imshow("Image", np.concatenate((no_skin, red, green, blue),axis=1))                                                                      
cv2.waitKey()


