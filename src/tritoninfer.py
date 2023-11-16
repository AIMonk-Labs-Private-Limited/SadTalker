import os
import shutil

import torch

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
import cv2

FACERENDER_BATCH_SIZE = 2

GLBL_RESIZE_ASPECT_RATIO_DIMS=1024

## resize function
def resize_aspect_ratio(img_frame):
    h,w=img_frame.shape[:2]
    large_dims=max(h,w)
    if large_dims>GLBL_RESIZE_ASPECT_RATIO_DIMS:
        ratio=GLBL_RESIZE_ASPECT_RATIO_DIMS/large_dims
        new_h,new_w=(GLBL_RESIZE_ASPECT_RATIO_DIMS,int(w*ratio)) if large_dims==h else (int(h*ratio),GLBL_RESIZE_ASPECT_RATIO_DIMS)
        return cv2.resize(img_frame,(int(new_w),int(new_h)),interpolation=cv2.INTER_LINEAR)
    else:
        return img_frame

class SadTalkerInfer():
    
    def __init__(self, config_dir, checkpoint_dir, size=256, preprocess="full",  
                 old_version=True, device="cuda:0"):
        '''
            config_dir: path to the config directory
            checkpoint_dir: path to the checkpoint directory
            size: the image size of the facerender
            preprocess: the preprocess method of the input image.
                Choose from ['crop', 'extcrop', 'resize', 'full', 'extfull']
            old_version: use the pth other than safetensor version
            device: device to run the model on
        '''
        self.size = size
        self.device = device
        self.preprocess = preprocess
        sadtalker_paths = init_path(
            checkpoint_dir, config_dir, size, old_version, preprocess
        )
        
        #init model
        self.preprocess_model = CropAndExtract(sadtalker_paths, device)
        self.audio_to_coeff = Audio2Coeff(sadtalker_paths,  device) 
        self.animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
    
    def create_temp_resize_image(self,image_source,source_image_flag):
        ## create resized temp image
        pic_name = os.path.splitext(os.path.split(image_source)[-1])[0] 
        file_path=os.path.join(os.path.dirname(image_source),pic_name+'_videotemp.mp4')
        if not os.path.isfile(image_source):
            raise ValueError('image_source must be a valid path to video/image file')
        elif image_source.split('.')[-1] in ['jpg', 'png', 'jpeg']:
            
            
            img_temp=cv2.imread(image_source)
            img_temp=resize_aspect_ratio(img_temp)
            print("image resized to: ",img_temp.shape)
            cv2.imwrite(image_source,img_temp)
            print("Temp image created from image",image_source)
            return image_source
            
        else:
            
            # loader for videos
            video_stream = cv2.VideoCapture(image_source)
            fps = video_stream.get(cv2.CAP_PROP_FPS)
            while 1:
                still_reading, frame = video_stream.read()
                if not still_reading:
                    video_stream.release()
                    break
                video_stream.release() 
                break
                ## resize the video
                
            full_img=resize_aspect_ratio(frame)
            frame_h = full_img.shape[0]
            frame_w = full_img.shape[1]
            print("video frame resized to: ",(frame_w,frame_h))
            video_stream_2 = cv2.VideoCapture(image_source)
            fps = video_stream_2.get(cv2.CAP_PROP_FPS)
            out_tmp = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (frame_w, frame_h))
            while 1:
                still_reading, frame = video_stream_2.read()
                if not still_reading:
                    video_stream_2.release()
                    break
                frame=resize_aspect_ratio(frame)
                out_tmp.write(frame)
            print("Temp video created from video",file_path)
            out_tmp.release()
            return file_path

class SadTalkerInfer():
    
    def __init__(self, config_dir, checkpoint_dir, size=256, preprocess="full",  
                 old_version=True, device="cuda:0"):
        '''
            config_dir: path to the config directory
            checkpoint_dir: path to the checkpoint directory
            size: the image size of the facerender
            preprocess: the preprocess method of the input image.
                Choose from ['crop', 'extcrop', 'resize', 'full', 'extfull']
            old_version: use the pth other than safetensor version
            device: device to run the model on
        '''
        self.size = size
        self.device = device
        self.preprocess = preprocess
        sadtalker_paths = init_path(
            checkpoint_dir, config_dir, size, old_version, preprocess
        )
        
        #init model
        self.preprocess_model = CropAndExtract(sadtalker_paths, device)
        self.audio_to_coeff = Audio2Coeff(sadtalker_paths,  device) 
        self.animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
    
    @torch.no_grad()
    def infer(self, driven_audio, image_source, image_path, result_dir, 
              video_as_source=False, ref_eyeblink=None, 
              ref_pose=None, pose_style=0, batch_size=FACERENDER_BATCH_SIZE, expression_scale=1, 
              input_yaw=None, input_pitch=None, input_roll=None, enhancer=False, 
              background_enhancer=None, still=False):
        '''
        Args:
            driven_audio: path to the audio file
            image_source: The source of image. It could be image or video
            image_path: Path to the image to save/load source image
            result_dir: path to the result directory
            video_as_source: Whether to use the middle frame from video as the source image
            ref_eyeblink: path to the reference eyeblink video
            ref_pose: path to the reference pose video
            pose_style: input pose style from [0, 46)
            batch_size: the batch size of facerender
            expression_scale: the scale of the expression from (0,1]
            input_yaw: the input yaw degree of the user
            input_pitch: the input pitch degree of the user
            input_roll: the input roll degree of the user
            enhancer: whether to use the enhancer (if True gfpgan enhancer will be used)
            still: can crop back to the original videos for the full body aniamtion
        '''
        enhancer = "gfpgan" if enhancer else None
        #crop image and extract 3dmm from image
        first_frame_dir = os.path.join(result_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        print("Started creating temp image/video")
        image_source=self.create_temp_resize_image(image_source,source_image_flag=False)
        print("done temp image/video ",image_source)
        print('3DMM Extraction for source image')
        first_coeff_path, crop_pic_path, crop_info =  self.preprocess_model.generate(
            image_source, first_frame_dir, self.preprocess, 
            video_as_source=video_as_source, video_image_path=image_path,
            source_image_flag=False, pic_size=self.size
        )
        
        if first_coeff_path is None:
            print("Can't get the coeffs of the input")
            return
        
        if ref_eyeblink is not None:
            ref_eyeblink_videoname = os.path.splitext(
                os.path.split(ref_eyeblink)[-1]
            )[0]
            ref_eyeblink_frame_dir = os.path.join(result_dir, ref_eyeblink_videoname)
            os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing eye blinking')
            ref_eyeblink_coeff_path, _, _ =  self.preprocess_model.generate(
                ref_eyeblink, ref_eyeblink_frame_dir, self.preprocess, 
                source_image_flag=False
            )
        else:
            ref_eyeblink_coeff_path=None
            
        if ref_pose is not None:
            if ref_pose == ref_eyeblink: 
                ref_pose_coeff_path = ref_eyeblink_coeff_path
            else:
                ref_pose_videoname = os.path.splitext(
                    os.path.split(ref_pose)[-1]
                )[0]
                ref_pose_frame_dir = os.path.join(result_dir, ref_pose_videoname)
                os.makedirs(ref_pose_frame_dir, exist_ok=True)
                print('3DMM Extraction for the reference video providing pose')
                ref_pose_coeff_path, _, _ =  self.preprocess_model.generate(
                    ref_pose, ref_pose_frame_dir, self.preprocess, 
                    source_image_flag=False
                )
        else:
            ref_pose_coeff_path=None
            
        #audio2ceoff
        batch = get_data(
            first_coeff_path, driven_audio, self.device, ref_eyeblink_coeff_path, 
            still=still
        )
        coeff_path = self.audio_to_coeff.generate(
            batch, result_dir, pose_style, ref_pose_coeff_path
        )
        
        #coeff2video
        data = get_facerender_data(
            coeff_path, crop_pic_path, first_coeff_path, driven_audio, 
            batch_size, input_yaw, input_pitch, input_roll, 
            expression_scale=expression_scale, still_mode=still, 
            preprocess=self.preprocess, size=self.size
        )
        
        result = self.animate_from_coeff.generate(
            data, result_dir, image_path, crop_info, 
            enhancer=enhancer, background_enhancer=background_enhancer, 
            preprocess=self.preprocess, img_size=self.size
        )
        
        return result
        
        