########### Assumption #############:
# 1. BackGround color is of same color throughout,
#    with no objects or noise present.
#
####################################

##### Alternate:
# 1. Cloth color, texture, and face classification technique and then
#	comparing the image
#
#############

import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i1", "--image1", required=True,
	help="path to input image 1")
ap.add_argument("-i2", "--image2", required=True,
	help="path to input image 2")
ap.add_argument("-path", "--path", required=True,
	help="path to graph prototbuf inceptionv3 tensorflow")
args = vars(ap.parse_args())

img1 = args["image1"]
img2 = args["image2"]
path = args["path"]

# Mean Normalization and Blurring of an Image
# is internally handled by Inception Archtiecture

'''
img1 = img1 - np.mean(img1, axis=0)
img2 = img2 - np.mean(img2, axis=0)

if(args["blur"]):
	img1 = cv2.GaussianBlur(img1, (BLUR, BLUR), 0)
	img2 = cv2.GaussianBlur(img2, (BLUR, BLUR), 0)
'''


path = '/home/aditya/Documents/classify_image_graph_def.pb'
list_images = [img1, img2]

#Suppress the Warning
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Intilaizing the Graph
def create_graph():
	with gfile.FastGFile(path,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')


# Extracting the Features from last Pool Layer of Inception v3 
def extract_features(list_images):
	nb_features = 2048
	features = np.empty((2,2048))

	create_graph()

	with tf.Session() as sess:
		next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

		for ind, image in enumerate(list_images):
		
			image_data = gfile.FastGFile(image, 'rb').read()
			#image_data = (image_data - image_data.mean(axis=0))/image_data.std(axis=0)
			predictions = sess.run(next_to_last_tensor,
				{'DecodeJpeg/contents:0': image_data})
			features[ind,:] = np.squeeze(predictions)
	

	return features


features = extract_features(list_images)
#print features.shape
features0 = features[0].ravel()
features1 = features[1].ravel()
#print features0.shape

sess = tf.Session()

print "Similarity:", cosine_similarity(features0.reshape(1,-1), features1.reshape(1,-1))

# Uncomment for Knowing the Operations in Graph
#for i in tf.get_default_graph().get_operations():
#    print i.name

op = sess.graph.get_operations()
hh = [m.values() for m in op]

# print m.values()
# mixed/tower_2/pool:0' shape=(1, 35, 35, 192) 
# pool_1:0' shape=(1, 35, 35, 192) dtype=float32>,
'''for ind in hh:
	print ind
	print
'''
