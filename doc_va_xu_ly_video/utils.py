import cv2
import os
import numpy as np
import torch
import tqdm
from torchvision.transforms import transforms
import os
import cv2
import numpy as np

transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)
topil = transforms.Compose(
    [
        transforms.ToPILImage()
    ]
)
class ImproveVideo:
    def __init__(self, model=None, skip=2, size=None, batch=10, device="cuda"):
        self.model = model
        self.skip = skip
        self.size = size
        self.batch = batch
        self.device = device

    def improve_video(self, input_video_path, output_video_path):
        video_capture = cv2.VideoCapture(input_video_path)
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc(*'XVID'), fps, (1408,1152 ))
        count = 0
        list_image = []

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            if count % self.skip == 0:
                frame = np.transpose(frame,(2,0,1))
                list_image.append(torch.tensor(frame).unsqueeze(0))
                if len(list_image) == self.batch:
                    out =  self.pipeline(list_image, out)
                    list_image = []
            count += 1
        if len(list_image)!=0:
            out = self.pipeline(list_image, out)
        out.release()
        video_capture.release()

    def _postprocessing_pipeline(self, output):
        img = np.transpose(output.cpu().clamp_(0, 1).numpy(),(0,2,3,1))
        img = (img * 255.0).round().astype(np.uint8)
        return img

    def pipeline(self, image_list, video):
        input_data = torch.cat(image_list,dim = 0).to("cuda")
        self.model.net_g_ema.eval()
        # import pdb; pdb.set_trace()
        with torch.no_grad():
           output = self.model.net_g_ema(input_data)
        input_data = input_data.to('cpu')
        output_list = self._postprocessing_pipeline(output)
        video  = self._save_frame(output_list, video)
        return video

    def _save_frame(self, frames, video):
        for i in frames.shape[0]:
            video.write(frames[i])
        return video
    
    def improve_a_folder(self, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for filename in tqdm.tqdm(os.listdir(input_folder)):
            if filename.endswith('.mp4'):
                input_video_path = os.path.join(input_folder, filename)
                output_video_path = os.path.join(output_folder, filename)
                self.improve_video(input_video_path, output_video_path)

  # Replace YourModelClass with the actual class of your model
reader = ImproveVideo(model=model)
reader.improve_a_folder("/content/drive/MyDrive/Input", "/content/Output3")
