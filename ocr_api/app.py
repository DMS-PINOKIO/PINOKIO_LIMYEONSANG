import os
import json
from flask.globals import request
from flask.helpers import flash
import numpy as np
from flask import Flask, render_template
from werkzeug.utils import redirect, secure_filename

from constants import constant
from libs import init, create_contour, rotate_cut, compare_char_size, run_ocr
    

def load_model(model_name):
    from tensorflow.keras.models import load_model
    model = load_model(model_name)
    return model

model = load_model(os.path.join(os.path.abspath(
    os.path.dirname(__file__)), 'kor_ckpt.h5'))


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = constant.UPLOAD_FOLDER

def allowed_file(filename):
    return filename.rsplit('.', 1)[1].lower() in constant.ALLOWED_EXTENSIONS

def get_test_img(path):
    from tensorflow.keras.preprocessing.image import img_to_array
    import PIL.Image as Img

    img = Img.open(path).convert('RGB')
    img = img.resize((28, 28))
    x = img_to_array(img)
    x = x.reshape((1,)+x.shape)
    return x


def decode_predict(result, labels):
    import numpy as np
    max = np.max(result)
    index = np.where(result == max)
    return labels[index[1][0]]

def predict(img_path):
    labels = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ']
    test_img = get_test_img(img_path)
    result = model.predict(test_img)
    decode = decode_predict(result, labels=labels)
    return decode

def main(img_path):
    height, width, _, img_thresh, contours = init.init(img_path)
    
    contours_dict = create_contour.create_contour(contours)

    possible_contours = compare_char_size.compare_char_size(contours_dict)

    def find_chars(contour_list):
        matched_result_idx = [] 

        for d1 in contour_list: 
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']: 
                    continue

                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])
                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2) 
                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']])) 

                if dx == 0: 
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx)) 

                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h']) 
                width_diff = abs(d1['w'] - d2['w']) / d1['w'] 
                height_diff = abs(d1['h'] - d2['h']) / d1['h'] 

                if distance < diagonal_length1 * constant.MAX_DIAG_MULTIPLYER \
                and angle_diff < constant.MAX_ANGLE_DIFF and area_diff < constant.MAX_AREA_DIFF \
                and width_diff < constant.MAX_WIDTH_DIFF and height_diff < constant.MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx']) 

            matched_contours_idx.append(d1['idx'])

            if len(matched_contours_idx) < constant.MIN_N_MATCHED: 
                continue

            matched_result_idx.append(matched_contours_idx) 

            unmatched_contour_idx = [] 
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx: 
                    unmatched_contour_idx.append(d4['idx']) 

            unmatched_contour = np.take(possible_contours, unmatched_contour_idx) 

            recursive_contour_list = find_chars(unmatched_contour)

            for idx in recursive_contour_list:
                matched_result_idx.append(idx) 

            break

        return matched_result_idx

    
    result_idx = find_chars(possible_contours)
    
    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    plate_imgs = rotate_cut.rotate_cut(matched_result, img_thresh, width, height)

    result_chars = run_ocr.run_ocr(plate_imgs)
    print(result_chars)

    return result_chars

@app.route('/braille', methods=['GET'])
def receive_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    f = request.files['file']
    if f.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if f and allowed_file(f.filename):
        f.save(os.path.join(
            app.config['UPLOAD_FOLDER']+secure_filename(f.filename)))
        char = predict(os.path.join(
            app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        return json.dumps(char, ensure_ascii=False)


@app.route('/ocr', methods=['GET'])
def receive():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    f=request.files['file']
    if f.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if f and allowed_file(f.filename):
        f.save(os.path.join(app.config['UPLOAD_FOLDER']+secure_filename(f.filename)))
        ocr_result = main(os.path.join(app.config['UPLOAD_FOLDER']+secure_filename(f.filename)))
    
    return json.dumps(ocr_result, ensure_ascii=False)

if __name__ == '__main__':
    app.run()