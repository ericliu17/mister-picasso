from flask import url_for, render_template
from app import app
import os
from random import choice
global random_combo_img
import json


base_imgs = os.listdir(os.path.join(app.static_folder, 'img/base'))
style_imgs = os.listdir(os.path.join(app.static_folder, 'img/style'))
combo_imgs = os.listdir(os.path.join(app.static_folder, 'img/combo'))


@app.route('/')
@app.route('/index')
def index():
    global random_combo_img
    random_combo_img = choice(combo_imgs)
    img_url = url_for('static',
                      filename=os.path.join('img/combo', random_combo_img))
    return render_template('index.html', title='Home', img_url=img_url)


@app.route('/zoom')
def zoom():
    global random_combo_img
    imgs = random_combo_img[:-4].split('_')
    base_img = imgs[0] + '.jpg'
    style_img = imgs[1] + '.jpg'
    base_img_url = url_for('static',
                           filename=os.path.join('img/base', base_img))
    style_img_url = url_for('static',
                           filename=os.path.join('img/style', style_img))
    img_url = url_for('static',
                      filename=os.path.join('img/combo', random_combo_img))
    # base_img_url = base_img
    # style_img_url = style_img
    # img_url = combo_imgs
    return render_template('zoom.html', title='Zoomed',
                           base_img_url=base_img_url,
                           style_img_url=style_img_url,
                           img_url=img_url)


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
