import matplotlib.pyplot as plt
import numpy as np

def visualizeData(width, height, px_vals):
	'''Takes the 1-D list of pixel intensities, makes them 
	rectangular, and displays them. The px_vals variable 
	should be a 1-D numpy array.'''

	img = np.reshape(255-px_vals, (height, width))
	img_plot = plt.imshow(img, cmap = 'gray') 
		#interpolation = 'nearest')
	plt.show()