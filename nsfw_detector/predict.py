#! python

import argparse
import json
from os.path import exists

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import base64
from PIL import Image
import keras
import io


IMAGE_DIM = 224   # required/default image dimensionality

def prepare_image(img, image_size):
	# resize the array (image) then PIL image
	im_resized = img.resize(image_size)
	img_array = keras.preprocessing.image.img_to_array(im_resized)
	img_array /= 255
	return img_array

def load_images(images_b64, image_size, verbose=True):
	'''
	Function for loading images into numpy arrays for passing to model.predict
	inputs:
		image_paths: list of image paths to load
		image_size: size into which images should be resized
		verbose: show all of the image path and sizes loaded
	
	outputs:
		loaded_images: loaded images on which keras model can run predictions
		loaded_image_indexes: paths of images which the function is able to process
	
	'''

	loaded_images = []

	for image_b64 in images_b64:
		try:
			image_bytes = base64.b64decode(image_b64)
			img = Image.open(io.BytesIO(image_bytes))
			loaded_images.append(prepare_image(img, image_size))
		except Exception as ex:
			raise ex

	return np.asarray(loaded_images)


def load_model(model_path):
	if model_path is None or not exists(model_path):
		raise ValueError("saved_model_path must be the valid directory of a saved model to load.")
	
	model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer},compile=False)
	return model


def classify(model, input_paths, input_bytes, image_dim=IMAGE_DIM):
	""" Classify given a model, input paths (could be single string), and image dimensionality...."""
	images = load_images(input_bytes, (image_dim, image_dim))
	probs = classify_nd(model, images)
	return dict(zip(input_paths, probs))


def classify_nd(model, nd_images):
	""" Classify given a model, image array (numpy)...."""

	model_preds = model.predict(nd_images)
	# preds = np.argsort(model_preds, axis = 1).tolist()
	
	categories = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

	probs = []
	for i, single_preds in enumerate(model_preds):
		single_probs = {}
		for j, pred in enumerate(single_preds):
			single_probs[categories[j]] = float(pred)
		probs.append(single_probs)
	return probs


def main(args=None):
	parser = argparse.ArgumentParser(
		description="""A script to perform NFSW classification of images""",
		epilog="""
		Launch with default model and a test image
			python nsfw_detector/predict.py --saved_model_path mobilenet_v2_140_224 --image_source test.jpg
	""", formatter_class=argparse.RawTextHelpFormatter)
	
	submain = parser.add_argument_group('main execution and evaluation functionality')
	submain.add_argument('--image_source', dest='image_source', type=str, required=True, 
							help='A directory of images or a single image to classify')
	submain.add_argument('--saved_model_path', dest='saved_model_path', type=str, required=True, 
							help='The model to load')
	submain.add_argument('--image_dim', dest='image_dim', type=int, default=IMAGE_DIM,
							help="The square dimension of the model's input shape")
	if args is not None:
		config = vars(parser.parse_args(args))
	else:
		config = vars(parser.parse_args())

	if config['image_source'] is None or not exists(config['image_source']):
		raise ValueError("image_source must be a valid directory with images or a single image to classify.")
	
	model = load_model(config['saved_model_path'])    
	image_preds = classify(model, config['image_source'], config['image_dim'])
	print(json.dumps(image_preds, indent=2), '\n')


if __name__ == "__main__":
	main()
