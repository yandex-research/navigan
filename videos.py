import numpy as np
import torch.nn.functional as F
import moviepy.editor as moviepy
import os
import torch
import cv2


def make_video(generator, zs, wd, file_dest, shift_from, shift_to, step=2,
               gif_seconds=3.5, interpolate=None,
               wd_deformate_arguments_builder=lambda shift_: [shift_, ]):
    assert 'avi' not in file_dest and 'mp4' not in file_dest

    clip_path = file_dest + '.avi'
    clip_path_mp4 = clip_path.replace('avi', 'mp4')
    if os.path.isfile(clip_path_mp4):
        return

    shifts = np.arange(shift_from, shift_to + step, step)
    
    imgs_total = []
    for shift in shifts:
        wd.deformate(*wd_deformate_arguments_builder(shift))
        with torch.no_grad():
            if interpolate is not None:
                imgs = F.interpolate(generator(zs), size=interpolate)
                imgs = (imgs.cpu().numpy().transpose((0, 2, 3, 1)) + 1) / 2
            else:
                imgs = (generator(zs).cpu().numpy().transpose((0, 2, 3, 1)) + 1) / 2
            imgs = np.concatenate(imgs, axis=1)
        imgs_total.append(imgs)
    wd.disable_deformation()
    
    imgs_total = np.stack(imgs_total, axis=0)
    
    imgs_total = (imgs_total * 255).astype('uint8')
    imgs_total = imgs_total[..., ::-1] # RGB -> BGR
    
    out = cv2.VideoWriter(clip_path,
                          cv2.VideoWriter_fourcc('M','J','P','G'),
                          len(imgs_total) / gif_seconds,
                          (imgs_total.shape[2], imgs_total.shape[1]))
        
    for img in imgs_total:
        out.write(img)
    for img in imgs_total[::-1]:
        out.write(img)

    out.release()

    avi_clip = moviepy.VideoFileClip(clip_path)
    avi_clip.write_videofile(clip_path_mp4, logger=None)
    os.remove(clip_path)
