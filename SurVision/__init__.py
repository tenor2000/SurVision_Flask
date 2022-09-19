from flask import Flask
from flask_sqlalchemy import SQLAlchemy
# from flask_bcrypt import Bcrypt
# from flask_login import LoginManager

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///user_log.db'
db = SQLAlchemy(app)



# attach username/userid later to folder name, for multiuser
app.config['UPLOAD_FOLDER'] = "SurVision/UPLOAD_FOLDER/"
# attach username/userid later to folder name, for multiuser
app.config['DOWNLOAD_FOLDER'] = "SurVision/DOWNLOAD_FOLDER/"
# Complete csv results go here
app.config['RESULTS'] = "SurVision/RESULTS_FOLDER/"

app.config['SURVEY_KEYS_FOLDER'] = "SurVision/SURVEY_KEYS_FOLDER/"

# 800 MB limit
app.config['MAX_CONTENT_LENGTH'] = 8 * 10000 * 10000

# Figure out a better way to generate
app.config['SECRET_KEY'] = 'b5c8fa0206ffd3ac3d3e27c9'

from SurVision import routes
