from flask import render_template, request, redirect, flash, jsonify
from werkzeug import secure_filename
from app import app
import os
import model.mister_picasso as mp


style_path = os.path.join(app.static_folder, 'img/style/')
combo_path = os.path.join(app.static_folder, 'img/combo/')
upload_base_path = 'static/uploads/base/'
upload_combo_path = 'static/uploads/combo/'
model_base_path = os.path.join(app.static_folder, 'uploads/base/')
model_combo_path = os.path.join(app.static_folder, 'uploads/combo/')


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
            base_img.save(model_base_path + base_file)

        base_img_url = upload_base_path + base_file
        return render_template('result.html', base_img_url=base_img_url)


@app.route('/generate', methods=['GET'])
def generate():
    weights_path = '../vgg16_weights.h5'
    # dimensions of the generated picture
    img_width = 128
    img_height = 128

    base_img = request.args.get('base_img')
    style_img = request.args.get('style_img')
    print 'Base file:', base_img
    print 'Style file:', style_img

    combo_img_url = mp.main(weights_path,
                            model_base_path, base_img,
                            style_path, style_img,
                            model_combo_path,
                            img_width, img_height,
                            iterations=10)

    combo_img = combo_img_url.split("/")[-1]
    return jsonify(url=upload_combo_path + combo_img)


def allowed_file(filename):
    return filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']
