# import sys
# sys.path.insert(1, '../model')
from flask import render_template, Blueprint, request, redirect, url_for, send_from_directory
from werkzeug import secure_filename
from app import app
import os
from scipy.misc import imread, imresize, imsave
import model.mister_picasso
# from model import model


base_path = os.path.join(app.static_folder, 'img/base')
style_path = os.path.join(app.static_folder, 'img/style')
combo_path = os.path.join(app.static_folder, 'img/combo')

@app.route('/')
@app.route('/index')
def index():
    combo_imgs = os.listdir(combo_path)
    return render_template('index.html', title='Home', combo_imgs=combo_imgs)


@app.route('/zoom')
def zoom():
    return render_template('zoom.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/submit')
def submit():
    style_imgs = os.listdir(style_path)
    return render_template('submit.html', style_imgs=style_imgs)


# Define the blueprint: 'generate', set its url prefix: app.url/generate
mod_generate = Blueprint('generate', __name__, url_prefix='/generate')


@app.route('/generate', methods=['GET', 'POST'])
def generate():
    base_path = 'uploads/base/'
    style_path = '/static/img/style/'
    weights_path = '../vgg16_weights.h5'

    # dimensions of the generated picture
    img_width = 128
    img_height = 128

    if request.method == 'POST':
        if 'imagefile' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['imagefile']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['BASE_UPLOAD_FOLDER'],
                                     filename)
            file.save(upload_path)

        # main(weights_path, base_path, base_file, style_path, style_file,
        #      img_width, img_height)

        base_img_url = url_for('static', filename=base_path + filename)
        return render_template('result.html', base_img_url=base_img_url)


def allowed_file(filename):
    return filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
