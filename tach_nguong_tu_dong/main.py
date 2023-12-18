import numpy as np 
import os
import cv2
import matplotlib.pyplot as plt

def histogram(image:np.array,num_value = 9):
    hist = [0]*num_value
    image  = image.reshape(-1).tolist()
    for value in image:
        hist[value]+=1
    return np.array(hist)

def tg(histogram_list:list,num_value):
    t_g_list = [histogram_list[0]]
    for i in range(1,num_value):
        t_g_list.append(histogram_list[i] + t_g_list[-1])
    return t_g_list
       
def mg(t_g_list,histogram_list):
    m_g_list = []
    for g in range(len(t_g_list)):
        momen_hi = 0
        for i in range(g+1):
            momen_hi+=i*histogram_list[i]
        m_g = momen_hi/t_g_list[g]
        m_g_list.append(m_g)
    return m_g_list
def Ag(t_g_list,P):
    t_g_array  = np.array(t_g_list)
    # a_g_array = 
    return t_g_array/(P-t_g_array)
def Bg(m_g_list,G):
    m_g_array = np.array() 
def tach_nguong_tu_dong(image,G , P):
    histo = histogram(image,G)
    t_g_list = tg(histo,G)
    m_g_list = mg(t_g_list,histo)
    a_g_array = Ag(t_g_list,P)
    
    print(histo)
    print(t_g_list)
    print(m_g_list)
    print(a_g_array)
    
image = [
    [1,4,2,8,5,7],
    [4,2,8,5,7,1],
    [0,8,5,7,1,4],
    [0,0,7,1,4,2],
    [0,0,0,4,2,8],
    [0,0,0,0,8,5],
    [0,0,0,0,0,7],
    [2,3,2,5,4,4],
    [3,2,5,4,4,2],
    [1,5,4,4,2,3],
    [1,1,4,2,3,2],
    [1,1,1,3,2,5],
    [1,1,1,1,5,4],
    [1,1,1,1,1,4]
]
image = np.array(image)
G = 9
P = image.shape[0]*image.shape[1]

tach_nguong_tu_dong(image,G,P)

# plt.subplot(1,2,1)
# plt.bar(range(9), histo, width=1.0, color='blue') 
# plt.ylabel('Count')
# plt.xlabel('Pixel Value')
# plt.title('Pixel Value Histogram')

# plt.subplot(1,2,2)
# plt.bar(range(9), f_g_list, width=1.0, color='blue') 
# plt.ylabel('Count')
# plt.xlabel('Pixel Value')
# plt.title('Pixel Value Histogram')
# plt.show()