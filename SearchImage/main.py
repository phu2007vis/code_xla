import numpy as np 
import matplotlib.pyplot as plt
import cv2
import glob
import os
import tqdm
import pickle

def histogram(image:np.array):
    total_pixel = 1
    for value in image.shape:
        total_pixel *= value
    hist = [0]*256
    image  = image.reshape(-1).tolist()
    for value in image:
        hist[value]+=1
    return np.array(hist)/total_pixel

class SearchImage:
    def __init__(self,image_dir = r"Image",check_point_path="checkpoint.pkl") -> None:
        self.data = self.load_data(image_dir,check_point_path)

    def find_image(self,path,distance_type = "euclid",show = True):

        if not os.path.exists(path):
            print(f"File {path} not exists")

        histograms = self.data.values()
        image = cv2.imread(path)
        histogram_search_image = histogram(image)
        distances = [getattr(self,distance_type)(hist,histogram_search_image)for hist in histograms]
        min_distance_index = np.argmin(np.array(distances))
        path_image_fitest = list(self.data.keys())[min_distance_index]
        self.show(path,path_image_fitest)

    @staticmethod
    def show(path1,path2):
        image1 = cv2.resize(cv2.imread(path1),(400,400))
        image2 = cv2.resize(cv2.imread(path2),(400,400))
        cv2.imshow("first image",image1)
        cv2.imshow("recived iamge",image2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def euclid(vector1,vector2):
        vector1 = np.array(vector1).reshape(-1)
        vector2 = np.array(vector2).reshape(-1)
        assert(vector1.shape==vector2.shape)
        sub_vector = np.subtract(vector1,vector2)
        pow_vetor  = pow(sub_vector,2)
        return np.sqrt(np.sum(pow_vetor))
    
    def find_change(self,list_image:list,data:dict):
        keys = list(data.keys())
        change = False
        for key in keys:
            if key not in list_image:
                change = True
                data.pop(key)
                print(f"Remove a old file {key}")
        for image_path in list_image:
            if image_path not in keys:
                change = True
                image = cv2.imread(image_path)
                histogram_data = histogram(image)
                data[image_path] = histogram_data
                print(f"Load a new file{image_path}")
        return data,change

    def load_data(self,image_dir = r"Image",check_point_path="checkpoint.pkl"):
        if os.path.exists(check_point_path):
            with open(check_point_path,"rb") as f:
                data = pickle.load(f)
            list_image = glob.glob(os.path.join(image_dir,"*"))
            data,change = self.find_change(list_image,data)
            if change:
                with open(check_point_path,"wb") as f:
                    pickle.dump(data,f)
                print(f"Saved new change data at {check_point_path}")
        else:
            histogram_datas = []
            list_image = glob.glob(os.path.join(image_dir,"*"))
            for image_path in tqdm.tqdm(list_image):
                image = cv2.imread(image_path)
                histogram_data = histogram(image)
                histogram_datas.append(histogram_data)
            print("Read data succed")
            data = dict(zip(list_image,histogram_datas))
            with open(check_point_path,"wb") as f:
                pickle.dump(data,f)
            print(f"Saved data at {check_point_path}")
        return data

search_image = SearchImage()
search_image.find_image(r"C:\Users\phuoc\OneDrive\Pictures\bai_tap.jpg")