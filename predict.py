from mmdet.apis import init_detector, inference_detector, show_result_pyplot

class PredictFunc():
    def __init__(self,image_path):
        self.image_path=image_path
        self.checkpoint_file = 'description.pth'
        self.config_file = 'default_runtime.py'
        self.device='cpu'
        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)
        # image_path = 'sample_images/sample_image.png'
    
    def main_inf(self):
        result = inference_detector(self.model, self.image_path)
        print(result)
        show_result_pyplot(self.model, self.image_path , result, out_file = 'modelOpImage' + "/"+self.image_path.split('/')[1].split('.')[0]+"model_out.png")
        return result
