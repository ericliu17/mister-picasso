from flask import Flask, render_template
import os

app = Flask(__name__)

from app import views

# Accepted extensions
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'gif'])
