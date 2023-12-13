import cv2 
import numpy as np
import matplotlib.pyplot as plt


path = "quang_cao.jpg"
rgb_image = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
r,g,b = cv2.split(rgb_image)

path = "quang_cao.jpg"
image = cv2.imread(path)

def count_value(image, value):
    count = np.sum(np.where(image == value, 1, 0))
    return count

hist = [count_value(image, value) for value in range(256)]
print(hist)
# print(hist)
# r_hist =np.array([count_value(r, value) for value in range(256)])
# g_hist =np.array([count_value(g, value) for value in range(256)])
# b_hist =[count_value(b, value) for value in range(256)]

# plt.subplot(1,4,1)
# plt.bar(range(256), hist, width=1.0, color='gray') 
# plt.ylabel('Count')
# plt.xlabel('Pixel Value')
# plt.title('Pixel Value Histogram')

# plt.subplot(1,4,2)
# plt.bar(range(256), r_hist, width=1.0, color='red')  
# plt.ylabel('Count')
# plt.xlabel('Pixel Value') 
# plt.title('Red hist')

# plt.subplot(1,4,3)
# plt.bar(range(256), b_hist, width=1.0, color='blue')  
# plt.ylabel('Count')
# plt.xlabel('Pixel Value')
# plt.title('Blue hist')


# plt.subplot(1,4,4)
# plt.bar(range(256), g_hist, width=1.0, color='green')  
# plt.ylabel('Count')
# plt.xlabel('Pixel Value')
# plt.title('Green hist')

# plt.show()
