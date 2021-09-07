import cv2
from constants import constant
import numpy as np
    

def rotate_cut(matched_result, img_thresh, width, height):
    plate_imgs = []
    plate_infos = []

    for _, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2 
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * constant.PLATE_WIDTH_PADDING 

        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h'] 

        plate_height = int(sum_height / len(sorted_chars) * constant.PLATE_HEIGHT_PADDING) 

        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']  
        triangle_hypotenus = np.linalg.norm( 
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )
        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus)) 

        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0) 

        img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height)) 

        img_cropped = cv2.getRectSubPix( 
            img_rotated, 
            patchSize=(int(plate_width), int(plate_height)), 
            center=(int(plate_cx), int(plate_cy))
        )

        if img_cropped.shape[1] / img_cropped.shape[0] < constant.MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < constant.MIN_PLATE_RATIO > constant.MAX_PLATE_RATIO:
            continue
        
        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })

    return plate_imgs