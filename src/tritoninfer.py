import os
import shutil

import torch

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

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
              ref_pose=None, pose_style=0, batch_size=2, expression_scale=1, 
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
        
        