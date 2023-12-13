import numpy as np
import cv2

def thresholding(image,max_value = 255,k = 155):
    return max_value/(1+ np.exp(- k + image ))


im = cv2.imread(r"quang_cao.jpg",cv2.IMREAD_GRAYSCALE)
im = cv2.resize(im,(440,440))
cv2.imshow("image",im)
cv2.waitKey(0)
thresh = thresholding(im)
cv2.imshow("image thresh",thresh)
cv2.waitKey(0)  
cv2.destroyAllWindows()

