from matplotlib import pyplot as plt
import numpy as np

img = np.array([[1, 0], [0, 1]])
plt.imshow(img, cmap='gray')

plt.savefig("image_features.png", bbox_inches='tight', pad_inches=0.1)