import cv2
import os
import numpy as np
import torch

class ImproveVideo:
    def __init__(self, model=None, skip=1,size = None,batch = 6,device = "cuda"):
        self.model = model.to(device)
        self.skip = skip
        self.size = size
        self.batch = batch
        self.device = device

    def improve_video(self, input_video_path, output_video_path):
        video_capture = cv2.VideoCapture(input_video_path)
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_video_path, -1, fps, (frame_width, frame_height))
        count = 0
        list_image = []
        while True:
            ret, frame = video_capture.read()
            
            if not ret:
                break
            if count % self.skip == 0:

                if self.size:
                    frame = cv2.resize(frame,self.size)
                list_image.append(frame)
                if len(list_image)==self.batch:
                    self._save_frame(list_image, out)
            count += 1
        if len(list_image):
            self._save_frame(list_image, out)
        video_capture.release()
        out.release()
    def _preprocessing_pipeline(self,image:np.array):
        image = image.astype(np.float32) / 255.0
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)
        return image
    def pipline(self,image_list):
        input = self.preprocessing_pipeline(image_list)
        output = self.model(input)
        list_image = self.possprocessing_pipeline(output)
        self._save_frame(list_image)
    def preprocessing_pipeline(self,list_image:list):
        data = []
        for image in list_image:
            data.append(self._preprocessing_pipeline(image))
        return torch.cat(data,dim = 0)
    def possprocessing_pipeline(self,output):
        output_list = []
        for img_index in range(output.shape[0]):
            img = output[img_index, :, :, :]
            img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))
            img = (img * 255.0).round().astype(np.uint8)
            output_list.append(img)
    def _save_frame(self, frames: list, video):
        for frame in frames:
            video.write(frame)

    def improve_a_folder(self, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for filename in os.listdir(input_folder):
            if filename.endswith('.mp4'):
                input_video_path = os.path.join(input_folder, filename)
                output_video_path = os.path.join(output_folder, filename)
                self.improve_video(input_video_path, output_video_path)

# Corrected folder names in the instantiation of ImproveVideo
reader = ImproveVideo().improve_a_folder(r"Input", r"Output")
