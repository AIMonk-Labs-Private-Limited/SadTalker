import os
from math import ceil
from scipy.spatial import ConvexHull
import torch
import torch.nn.functional as F
import cv2
import imageio
import imageio_ffmpeg
import numpy as np
from tqdm import tqdm 
from skimage import img_as_ubyte

# 2min 25fps video chunk size -> 2 * 60 (seconds) * 25 frames
RENDERING_CHUNK_SIZE = 2 * 60 * 25

def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new

def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).type_as(pred).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, 1) * 3 - 99
    return degree

def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat

def keypoint_transformation(kp_canonical, he, wo_exp=False):
    kp = kp_canonical['value']    # (bs, k, 3) 
    yaw, pitch, roll= he['yaw'], he['pitch'], he['roll']      
    yaw = headpose_pred_to_degree(yaw) 
    pitch = headpose_pred_to_degree(pitch)
    roll = headpose_pred_to_degree(roll)

    if 'yaw_in' in he:
        yaw = he['yaw_in']
    if 'pitch_in' in he:
        pitch = he['pitch_in']
    if 'roll_in' in he:
        roll = he['roll_in']

    rot_mat = get_rotation_matrix(yaw, pitch, roll)    # (bs, 3, 3)

    t, exp = he['t'], he['exp']
    if wo_exp:
        exp =  exp*0  
    
    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    # keypoint translation
    t[:, 0] = t[:, 0]*0
    t[:, 2] = t[:, 2]*0
    t = t.unsqueeze(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation 
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    return {'value': kp_transformed}

def create_semantic_chunks(target_semantics, chunk_size=RENDERING_CHUNK_SIZE):
    # target_semantics.shape -> (batch_size, frame_length, ?, ?)
    # total frames = batch_size * frame_length
    # chunk size is 2 mins -> 2 * 60 seconds with 25 fps
    total_frames = target_semantics.shape[1] * target_semantics.shape[0]
    no_chunks = ceil(total_frames / chunk_size)  # ceil division
    # adding 1 more chunk if there is any residual frames remaining
    # if total_frames % chunk_size:
    #     no_chunks += 1
    # we reduce chunk size because, in reality each sample would have 
    # batch_size number of frames, not 1
    reduced_chunk_size = ceil(chunk_size / target_semantics.shape[0])
    print(f"Number of chunks {no_chunks} with chunk size {reduced_chunk_size}")

    for i in range(0, no_chunks):
        start = i*reduced_chunk_size
        end = (i+1)*reduced_chunk_size
        # for last chunk, we might have less frames
        end = end if end < total_frames else total_frames
        yield target_semantics[:, start:end]

def save_chunk_video(predictions, chunk_index, video_save_dir, video_name, img_size, original_size):
    predictions_ts = torch.stack(predictions, dim=1)
    predictions_video = predictions_ts.reshape((-1,)+predictions_ts.shape[2:])
    
    video = []
    video_name = f"temp_{chunk_index}_" + video_name
    video_path = os.path.join(video_save_dir, video_name)
    for idx in range(predictions_video.shape[0]):
        image = predictions_video[idx]
        image = np.transpose(image.data.cpu().numpy(), [1, 2, 0]).astype(np.float32)
        image = img_as_ubyte(image)
        image =  cv2.resize(image, (img_size, int(img_size * original_size[1]/original_size[0]) ))
        video.append(image)
        
    imageio.mimsave(video_path, video,  fps=float(25))
    
    return video_path

def concat_chunks(chunk_videos, output_dir, video_name, total_frames, fps=25):
    output_video_path = os.path.join(output_dir, "temp_" + video_name)
    output_video = imageio.get_writer(output_video_path, fps=fps)
    
    frame_count = 0
    frame_count_exceed = False
    for chunk_video in chunk_videos:
        input_video = imageio.get_reader(chunk_video)
        for frame in input_video:
            # we only consider first N frames required according to audio
            # if it exceeds stop adding frames in it
            if frame_count > total_frames:
                frame_count_exceed = True
                break
            output_video.append_data(frame)
            frame_count += 1
            
        if frame_count_exceed:
            break
            
    output_video.close()
    
    return output_video_path

def make_animation(source_image, source_semantics, target_semantics,
                            generator, kp_detector, he_estimator, mapping, 
                            img_size, original_size, video_save_dir, video_name,
                            frame_num,
                            yaw_c_seq=None, pitch_c_seq=None, roll_c_seq=None,
                            use_exp=True, use_half=False):
    with torch.no_grad():
        chunk_videos = []

        kp_canonical = kp_detector(source_image)
        he_source = mapping(source_semantics)
        kp_source = keypoint_transformation(kp_canonical, he_source)

        for chunk_index, target_semantics_chunk in enumerate(create_semantic_chunks(target_semantics)):
            predictions = []
            print("Procssing Chunk: ", chunk_index)
            for frame_idx in tqdm(range(target_semantics_chunk.shape[1]), 'Face Renderer:'):
                # still check the dimension
                # print(target_semantics.shape, source_semantics.shape)
                target_semantics_frame = target_semantics_chunk[:, frame_idx]
                he_driving = mapping(target_semantics_frame)
                if yaw_c_seq is not None:
                    he_driving['yaw_in'] = yaw_c_seq[:, frame_idx]
                if pitch_c_seq is not None:
                    he_driving['pitch_in'] = pitch_c_seq[:, frame_idx] 
                if roll_c_seq is not None:
                    he_driving['roll_in'] = roll_c_seq[:, frame_idx] 

                kp_driving = keypoint_transformation(kp_canonical, he_driving)

                kp_norm = kp_driving
                out = generator(source_image, kp_source=kp_source, kp_driving=kp_norm)
                '''
                source_image_new = out['prediction'].squeeze(1)
                kp_canonical_new =  kp_detector(source_image_new)
                he_source_new = he_estimator(source_image_new) 
                kp_source_new = keypoint_transformation(kp_canonical_new, he_source_new, wo_exp=True)
                kp_driving_new = keypoint_transformation(kp_canonical_new, he_driving, wo_exp=True)
                out = generator(source_image_new, kp_source=kp_source_new, kp_driving=kp_driving_new)
                '''
                predictions.append(out['prediction'].cpu())

            chunk_videos.append(
                save_chunk_video(predictions, chunk_index, video_save_dir, 
                                 video_name, img_size, original_size)
            )
        print("Concating chunks...")
        video_path = concat_chunks(chunk_videos, video_save_dir, video_name, frame_num)
        #import pdb;pdb.set_trace()
    return video_path

class AnimateModel(torch.nn.Module):
    """
    Merge all generator related updates into single model for better multi-gpu usage
    """

    def __init__(self, generator, kp_extractor, mapping):
        super(AnimateModel, self).__init__()
        self.kp_extractor = kp_extractor
        self.generator = generator
        self.mapping = mapping

        self.kp_extractor.eval()
        self.generator.eval()
        self.mapping.eval()

    def forward(self, x):
        
        source_image = x['source_image']
        source_semantics = x['source_semantics']
        target_semantics = x['target_semantics']
        yaw_c_seq = x['yaw_c_seq']
        pitch_c_seq = x['pitch_c_seq']
        roll_c_seq = x['roll_c_seq']

        predictions_video = make_animation(source_image, source_semantics, target_semantics,
                                        self.generator, self.kp_extractor,
                                        self.mapping, use_exp = True,
                                        yaw_c_seq=yaw_c_seq, pitch_c_seq=pitch_c_seq, roll_c_seq=roll_c_seq)
        
        return predictions_video
