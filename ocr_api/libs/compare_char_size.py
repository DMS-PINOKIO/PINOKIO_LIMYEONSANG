from constants import constant

def compare_char_size(contours_dict):
    possible_contours = []

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h'] 
        ratio = d['w'] / d['h']

        if area > constant.MIN_AREA \
        and d['w'] > constant.MIN_WIDTH and d['h'] > constant.MIN_HEIGHT \
        and constant.MIN_RATIO < ratio < constant.MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)
    
 
    return possible_contours

