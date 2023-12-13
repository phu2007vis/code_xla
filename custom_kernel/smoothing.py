import cv2 
import numpy as np
import math
import matplotlib.pyplot as plt


def apply_kernel(image,kernel,kernel_value = None):
    kernel_size = kernel.shape[0]
    if kernel_value == None:
        kernel_value = kernel_size*kernel_size
    h_image, w_image,channel = image.shape
    vertical_n = int(math.floor(h_image -kernel_size+1))
    horiz_n = int(math.floor(w_image -kernel_size+1))

    new_image = np.zeros((vertical_n, horiz_n,channel))
    
    for v in range(vertical_n-1):
        v_start = v 
        v_end = v_start+kernel_size
        if v_end>h_image:
            break

        for h in range(horiz_n-1):
            h_start = h 
            h_end = h_start + kernel_size
            if h_end>w_image:
                break

            for c in range(channel):
                small_image = image[v_start:v_end, h_start:h_end, c]         
                new_image[v, h,c] = np.sum(np.multiply(small_image, kernel)) / (kernel_value)

    return np.asarray(new_image, dtype=np.uint8)

def mean_kernel_smooth(image, kernel_size=5):

    kernel = np.ones((kernel_size, kernel_size))
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        kernel = np.expand_dims(kernel, axis=-1)

    return apply_kernel(image,kernel)

def gausian_kernel(image):
    kernel = np.array(
        [[1,2,1],
        [2,4,2],
        [1,2,1]]
    )
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        kernel = np.expand_dims(kernel, axis=-1)
    return apply_kernel(image,kernel,16)

  
path = r"quang_cao.jpg"
img = cv2.cvtColor(cv2.resize(cv2.imread(path),(448,448)),cv2.COLOR_BGR2RGB)

kernel_sizes = [3,5,7]
images = []
plt.figure()
plt.imshow(img)
plt.title("origin image ")

for kernel_size in kernel_sizes:
    plt.figure()
    plt.imshow(mean_kernel_smooth(img,kernel_size))
    plt.title(f"Mean with kernel size {kernel_size}")
plt.figure()
plt.imshow(gausian_kernel(img))
plt.title("Gaussian ")
plt.show()

