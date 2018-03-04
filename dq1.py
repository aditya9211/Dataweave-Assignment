############## Assumption ##############:
# 1. Orientaion of image, block
#   size and shape is more or less the same, 
#   throughout the dataset
#
# 2. Rectangular Box enclosed with Serial No
#   and Voter ID lies just next to it.
########################################

##### Alternate:
# 1. AutoPredicting and correcting the id and serialNo with existing code
#
# 2. Eliminating the Small vertical lines enclosing the each rectangle, so no "I","|"etc
#################
import cv2
import argparse
from PIL import Image
import pytesseract as pyt
import os
import numpy as np

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-sno", "--sno", required=True,
	help="serial no to input image")
ap.add_argument("-b", "--blur", required=False, type = int, default = 5,
	help="serial no to input image")
args = vars(ap.parse_args())

# Reading Image and Parameters
gray = cv2.imread(args["image"])
sno = args["sno"]
BLUR = args["blur"]

#Line Detection
edges = cv2.Canny(gray,50,150,apertureSize = 3)
minLineLength=100
lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)

# Horizontal Line Width Checking in an Image
a,b,c = lines.shape
for i in range(a):
    if(lines[i][0][1] == lines[i][0][3]):
	x_cor1 = lines[i][0][0]	
	x_cor2 = lines[i][0][2]	
	break

i=0
k=0

# Detecting Any one Vertical Line
for i in range(a):
    k = k + 1
    if(lines[i][0][0] != lines[i][0][2]):
	continue

    cv2.line(gray, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imwrite('houghlines5.jpg',gray)
    break


#adjusting if any difference in 94 (x1) points
k = 0
ratio = (x_cor1-94)/10.0
ratio = int(ratio)

if(ratio < 10):
	ratio = 0

cropped = gray.copy()
cropped[:,:,:] = [255, 255, 255]


# Cropping the Desired Image
while(True):

	if(lines[i][0][3]+90+(400-ratio)*k > lines[i][0][1]):
		break
		
	cv2.line(gray, (94, lines[i][0][3]+0+(400-ratio)*k), (3203, lines[i][0][3]+0+(400-ratio)*k), (0, 0, 255), 3, cv2.LINE_AA)


	cv2.line(gray, (94, lines[i][0][3]+90+(400-ratio)*k), (3203, lines[i][0][3]+90+(400-ratio)*k), (0, 0, 255), 3, cv2.LINE_AA)

	cropped[lines[i][0][3]+0+17+(400-ratio)*k:lines[i][0][3]+90-5+(400-ratio)*k, 94:3203,:] = gray[lines[i][0][3]+0+17+(400-ratio)*k:lines[i][0][3]+90-5+(400-ratio)*k, 94:3203,:]


	k = k + 1	

cv2.imwrite('houghlines5555.jpg',gray) # Writing the Lined Image

cropped = cv2.GaussianBlur(cropped, (BLUR, BLUR), 0)
cv2.imwrite("Imgcrop.jpg", cropped) # Writing the Cropped Image


# Converting Image to fit in OCR
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)
#, 4, 6 config='digits -psm 7'
# config = "-psm 6"
text = pyt.image_to_string(Image.open(filename), config = "-psm 6", lang = 'eng')
os.remove(filename)
text = text.encode('utf-8')
print text

# Find the index
index = text.find(sno)

# Validating the Conditions I 788 |, I788 etc.
if (index!=-1 and text[index-1] == " " and text[index+len(sno)] == " " or index!=-1 and text[index-1] == "I" and text[index+len(sno)] == "I" or index!=-1 and text[index-1] == " " and text[index+len(sno)] == "I") :
    print "Found", sno, "in the string."
else:
    print "No", sno, "here!"
    exit()

ind = index+len(sno)

# Finding the Starting index of Voter ID
while(True):

	ind = ind + 1
	
	if((text[ind] == "I" and text[ind+1] == " ") or text[ind] == " ") or text[ind] == "|" or text[ind] == "l":
		continue
	else:
		break


length = 0
ind = ind - 1

stringlist = []

# Taking the Length as '10' as mentioned it has 196.8 billion combination
# surpassing total world's population which is 7.6 billion
# so sufficient to include all India Population Database
while(length < 10):
	ind = ind + 1
	if(text[ind] == " "):
		continue
	stringlist.append(text[ind])
	length = length + 1
	
# Printing the Resulting ID	
print ''.join(stringlist)




