import numpy as np
import matplotlib.pyplot as plt
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
def histogram(image:np.array,num_value = 9):
    total_pixel = 1
    for value in image.shape:
        total_pixel *= value
    hist = [0]*num_value
    image  = image.reshape(-1).tolist()
    for value in image:
        hist[value]+=1
    return np.array(hist),total_pixel

def tg(histogram_list:list,num_value):
    t_g = [histogram_list[0]]
    for i in range(1,num_value):
        t_g.append(histogram_list[i] + t_g[-1])
    return t_g
       
def fg(t_g_list,new_level,total_pixel):
    f_g_list = []
    for i in range(len(t_g_list)):
        f_g = max(0,round((t_g_list[i]/total_pixel)*(new_level)-1))
        f_g_list.append(f_g)
    return f_g_list
    

    
def can_bang_histogram(image:np.array,new_level = 6,num_value = 9):
    old_histogram,total_pixel = histogram(image)
    t_g_list = tg(old_histogram,num_value)
    print("hg: ",old_histogram)
    print("total pixel: ", total_pixel)
    print("tg:",t_g_list)
    f_g_list = fg(t_g_list,new_level,total_pixel)
    print("fg: ",f_g_list)
    return old_histogram,f_g_list

new_level = 6
histo,f_g_list = can_bang_histogram(np.array(image),new_level = new_level,num_value = 9)

plt.subplot(1,2,1)
plt.bar(range(9), histo, width=1.0, color='blue') 
plt.ylabel('Count')
plt.xlabel('Pixel Value')
plt.title('Pixel Value Histogram')

plt.subplot(1,2,2)
plt.bar(range(9), f_g_list, width=1.0, color='blue') 
plt.ylabel('Count')
plt.xlabel('Pixel Value')
plt.title('Pixel Value Histogram')
plt.show()