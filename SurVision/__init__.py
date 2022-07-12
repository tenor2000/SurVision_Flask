from flask import Flask

# from flask_sqlalchemy import SQLAlchemy
# from flask_bcrypt import Bcrypt
# from flask_login import LoginManager

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = "SurVision/UPLOAD_FOLDER/"
app.config['DOWNLOAD_FOLDER'] = "SurVision/DOWNLOAD_FOLDER/"
app.config['MAX_CONTENT_PATH'] = 16 * 1024 * 1024

# Figure out a better way to generate
app.config['SECRET_KEY'] = 'b5c8fa0206ffd3ac3d3e27c9'

from program import routes
