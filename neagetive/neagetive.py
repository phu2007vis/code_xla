import numpy as np
import cv2
import matplotlib.pyplot as plt

def negative(image):
    image = 255-image
    return image


def log_transform(image):
    # image = image/255
    c = 255 / np.log(1 + np.max(image)) 
    log_image = c * (np.log(image + 1)) 
    return np.asarray(log_image,dtype=np.uint8)

def constras(image,k = 0.5):
    image = image/255.0
    image = 1/(1+np.exp(10*(-image+k)))
    return (image*255).astype(np.uint8)

def thresh(image,k = 155):
    return np.where(image>k,255,0)

im = cv2.imread(r"quang_cao.jpg",cv2.IMREAD_GRAYSCALE)
im = cv2.resize(im,(440,440))
ret,thresh = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
im_add = cv2.add(im,np.full(shape=im.shape,fill_value=20,dtype=np.uint8))

def count_value(image, value):
    count = np.sum(np.where(image == value, 1, 0))
    return count




hist_origin = [count_value(im, value) for value in range(256)]
hist_negative = [count_value(negative(im.copy()), value) for value in range(256)]


for i,y in enumerate(range(2,10)):
    im2 = cv2.resize(im,(224,224))
    plt.subplot(1,8,i+1)
    plt.imshow(log_transform(im2.copy(),y/10),cmap="binary")
plt.show()

cv2.waitKey(0)
cv2.imshow("origin image",im)
cv2.imshow("negative image",negative(im.copy()))

plt.subplot(1,2,1)
plt.bar(range(256), hist_origin, width=1.0, color='blue') 
plt.ylabel('Count')
plt.xlabel('Pixel Value')
plt.title('Pixel Value Histogram')

plt.subplot(1,2,2)
plt.bar(range(256), hist_negative, width=1.0, color='red')  
plt.ylabel('Count')
plt.xlabel('Pixel Value') 
plt.title('Negative hist')
plt.show()
cv2.waitKey(0)

cv2.imshow("constras image",constras(im.copy()))
cv2.imshow("log image",log_transform(im.copy()))
cv2.imshow("thresh binary",thresh)
cv2.imshow("im add ",im_add)
cv2.waitKey(0)
cv2.destroyAllWindows()

