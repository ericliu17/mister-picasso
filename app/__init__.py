from flask import Flask, render_template
import os

app = Flask(__name__)

from app import views

# Path to the upload directory
BASE_UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads/base')
app.config['BASE_UPLOAD_FOLDER'] = BASE_UPLOAD_FOLDER

# Accepted extensions
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'gif'])

# # Sample HTTP error handling
# @app.errorhandler(404)
# def not_found(error):
#     return render_template('404.html'), 404
