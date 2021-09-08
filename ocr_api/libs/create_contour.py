import cv2

def create_contour(contours):
    contours_dict = []

    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)

        contours_dict.append({
        'contour':contour,
        'x':x,
        'y':y,
        'w':w,
        'h':h,
        'cx':x+(w/2),
        'cy':y+(h/2) 
    })

    return contours_dict
