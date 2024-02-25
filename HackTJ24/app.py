from flask import Flask, request, render_template_string, redirect, url_for, render_template, flash
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
import os
from werkzeug.utils import secure_filename
import cv2
from flask_cors import CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'  # Needed for flash messages
app.config['UPLOAD_FOLDER'] = '/Users/rp/Downloads/finexo-html/uploads'

class UpholdFileForm(FlaskForm):
    signature = FileField("Signature File")
    reference_signature = FileField("Reference Signature File")
    submit = SubmitField("Upload File")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/service', methods=['GET', 'POST'])
def service():
    if request.method == 'POST':
        redirect()
    return render_template('service.html')

@app.route('/why', methods=['GET', 'POST'])
def why():
    return render_template('why.html')

@app.route('/team', methods=['GET', 'POST'])
def team():
    return render_template('team.html')

@app.route('/about', methods=['GET', 'POST'])
def about():
    return render_template('about.html')

def augment_image(image):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))
    return rotated_image

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not load image at {image_path}. Check the file path and permissions.")
        return None
    resized_image = cv2.resize(image, (500, 100))  # Resize for consistency
    blur_image = cv2.GaussianBlur(resized_image, (5,5), 0)
    _, processed_image = cv2.threshold(blur_image, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return processed_image

def extract_features(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key = lambda x:x.distance)
    return matches

def check_forgery(matches, threshold=30):
    if len(matches) > threshold:
        return False  # Not a forgery
    else:
        return True

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UpholdFileForm()
    if request.method == 'POST':
        signature_file = form.signature.data
        reference_file = form.reference_signature.data

        # Save signature file
        if signature_file and allowed_file(signature_file.filename):
            signature_filename = secure_filename(signature_file.filename)
            signature_path = os.path.join(app.config['UPLOAD_FOLDER'], signature_filename)
            signature_file.save(signature_path)
        else:
            flash('Invalid file type for signature.')
            return redirect(request.url)

        # Save reference signature file
        if reference_file and allowed_file(reference_file.filename):
            reference_filename = secure_filename(reference_file.filename)
            reference_path = os.path.join(app.config['UPLOAD_FOLDER'], reference_filename)
            reference_file.save(reference_path)
        else:
            flash('Invalid file type for reference signature.')
            return redirect(request.url)

        # You can process the files as needed here
        processed_signature = preprocess_image(signature_path)
        processed_reference = preprocess_image(reference_path)
        
        augmented_signature = augment_image(processed_signature)
        augmented_reference = augment_image(processed_reference)

        kp1, des1 = extract_features(augmented_signature)
        kp2, des2 = extract_features(augmented_reference)

        matches = match_features(des1, des2)

        is_forgery = check_forgery(matches)
        if is_forgery:
            result_message = "The uploaded signature may be a forgery."
        else:
            result_message = "The uploaded signature is likely genuine."
        print(result_message)
        return render_template('service.html', form=form, output=result_message)

    return render_template('service.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
