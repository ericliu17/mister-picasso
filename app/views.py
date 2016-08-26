from flask import url_for, render_template
from app import app
import os


base_imgs = os.listdir(os.path.join(app.static_folder, 'img/base'))
style_imgs = os.listdir(os.path.join(app.static_folder, 'img/style'))
combo_imgs = os.listdir(os.path.join(app.static_folder, 'img/combo'))


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Home', combo_imgs=combo_imgs)


@app.route('/zoom')
def zoom():
    return render_template('zoom.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    print 'Reading the img'
    image = Image.open(request.files['imagefile'])
    results = predict(image)
    proc = proc_results(results, app_dict)
    # predictions = predict(imagefile)
    return render_template('result.html', data=proc)
