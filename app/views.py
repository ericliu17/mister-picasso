from flask import render_template, request, redirect, flash
from werkzeug import secure_filename
from app import app
import os
import model.mister_picasso as mp


base_path = os.path.join(app.static_folder, 'img/base/')
style_path = os.path.join(app.static_folder, 'img/style/')
combo_path = os.path.join(app.static_folder, 'img/combo/')
upload_base_path = 'static/uploads/base/'
upload_combo_path = 'static/uploads/combo/'


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


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':

        if 'imagefile' not in request.files:
            flash('No file part')
            return redirect(request.url)

        base_img = request.files['imagefile']

        if base_img.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # Check if the file is one of the allowed types/extensions
        if base_img and allowed_file(base_img.filename):
            base_file = secure_filename(base_img.filename)
            upload_path = os.path.join(app.config['BASE_UPLOAD_FOLDER'],
                                       base_file)
            base_img.save(upload_path)

        base_img_url = upload_base_path + base_file
        return render_template('result.html', base_img_url=base_img_url)


@app.route('/generate', methods=['GET'])
def generate():
    weights_path = '../vgg16_weights.h5'
    # dimensions of the generated picture
    img_width = 128
    img_height = 128

    base_file = request.args.get('base_img')
    style_file = request.args.get('style_img')
    print "Base file:", base_file
    print "Style file:", style_file

    # mp.main(weights_path, base_path, base_file, style_path, style_file,
        #  img_width, img_height)

    return render_template('result.html', combo_img_url=combo_img_url)


def allowed_file(filename):
    return filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
