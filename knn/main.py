import numpy as np
import os 
import glob 
import cv2
import tqdm
from sklearn.model_selection import train_test_split

# chuan hoa 0 - 1
def normalize(X:np.array):
    x_max = X.max()
    x_min = X.min()
    X = (X-x_min)/(x_max-x_min)
    return X
#trich xuat histogram
def histogram(image:np.array):
    total_pixel = 1
    for value in image.shape:
        total_pixel *= value
    hist = [0]*256
    image  = image.reshape(-1).tolist()
    for value in image:
        hist[value]+=1
    return np.array(hist)/total_pixel

#lay du lieu tu thu muc image
def load_data(image_dir ,label:int,size = (50,50)):
    histogram_datas = []
    #duyet ta ca anh trong thu muc
    list_image = glob.glob(os.path.join(image_dir,"*"))
    for image_path in tqdm.tqdm(list_image):
        image = cv2.imread(image_path)
        image = cv2.resize(image,size)
        histogram_data = histogram(image)
        histogram_datas.append(histogram_data)  
    #lay ra label cua toan bo thu muc
    labels = [label]*len(histogram_datas)
    return histogram_datas,labels
#sinh ra toan bo data
def generate_data(data_dir = "Image"):
    X = []
    Y = []
    name_class = []
    for i,image_dir in enumerate(glob.glob(os.path.join(data_dir,"*"))):
        #lay ra ten cua thu muc lam ten class
        name_class.append(image_dir.split(os.sep)[-1])
        x,y = load_data(image_dir,i)
        X.extend(x)
        Y.extend(y)
    #chuan hoa ve 0 - 1
    #tach thanh 2 tap 
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=8)
    return X_train, X_test, Y_train, Y_test,name_class
#model
class KNN:
    def __init__(self,X_train,Y_train,name_class,k_neighbor = 3) :
        self.X_train,self.Y_train = X_train,Y_train
        self.k_neighbor = k_neighbor
        self.name_class = name_class
        self.num_class = len(name_class)

    def predict_single(self,X,distance_type = "euclid"):
        #tinh khoang cach cua X voi toan bo du lieu da co
        distances = [self.euclid(X,x_train) for x_train in X_train]
        #gan khoang cach voi cac label
        distances_with_lable = zip(distances,self.Y_train)
        #xap xep
        distances_with_label = sorted(distances_with_lable,key=lambda x: x[0])
        #chon ra n label co  khoang cach nho nhat
        n_distances_min = distances_with_label[:self.k_neighbor]
        #tinh tuan suat
        tan_suat = [0]*self.num_class
        for distance, label in n_distances_min:
            tan_suat[label]+= 1      
        #label co tuan suat xuat hien nhieu nhat
        label = np.argmax(tan_suat)
        return label,self.name_class[label]  

    @staticmethod
    def euclid(vector1,vector2):
        vector1 = np.array(vector1).reshape(-1)
        vector2 = np.array(vector2).reshape(-1)
        assert(vector1.shape==vector2.shape)
        sub_vector = np.subtract(vector1,vector2)
        pow_vetor  = pow(sub_vector,2)
        return np.sqrt(np.sum(pow_vetor))
    
path_all_image = "Image"
X_train, X_test, Y_train, Y_test,name_class = generate_data(path_all_image)
model = KNN(X_train,Y_train,name_class)
print(model.predict_single(X_test[0]))
