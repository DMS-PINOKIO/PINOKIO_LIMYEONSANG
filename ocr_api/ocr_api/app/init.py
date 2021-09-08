import cv2

def init(img_path):
    img_og = cv2.imread(img_path)
    height, width, channel = img_og.shape

    gray = cv2.cvtColor(img_og, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(gray, ksize=(5,5), sigmaX=0)
    img_thresh = cv2.adaptiveThreshold(
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )

    contours, _ = cv2.findContours(
        img_thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    return height, width, channel, img_thresh, contours