import pandas as pd
from SurVision import app
from flask import render_template, redirect, url_for, flash, request, send_file
from werkzeug.utils import secure_filename
import os
from pdf2image import convert_from_path
from shutil import make_archive, move
import cv2
import scanner as scn
import time


def clean_directory(directory):
    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))

def extract_files():
    if request.method == 'POST':
        f = request.files['file']
        # clear old files
        try:
            # Clean directory before populating
            clean_directory(app.config['UPLOAD_FOLDER'])

            f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
            surveyPDFpath = "".join((app.config['UPLOAD_FOLDER'], f.filename))

            # Extract script here

            from extract import getSurveyFiles, rename_files
            conversion_path = app.config['DOWNLOAD_FOLDER']
            # clear old files
            for file in os.listdir(conversion_path):
                os.remove(os.path.join(conversion_path, file))

            images = convert_from_path(surveyPDFpath, dpi=150, fmt='png', grayscale=True,
                                       poppler_path=r"extract\poppler-21.11.0\Library\bin")
            getSurveyFiles(images, 1, 1, conversion_path)
            root_directory = os.getcwd()
            os.chdir(conversion_path)
            directory = os.listdir()

            rename_files(directory)
            # Extract script end

            # zip the directory with files
            make_archive('../survey_files', 'zip')
            # return to root directory
            os.chdir(root_directory)
            flash(f'{f.filename} was converted successfully!', category='success')

            return redirect(url_for('download_surveys_page'))

        except:
            flash(f'There was a problem with {f.name} file. Please make sure you are using a PDF.', category='danger')

### Scan Functions ###




####### ROUTES ########

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')


@app.route('/test/', methods=['POST'])
def test_button():
    scn.execute_survey_scanning('VA2021EV.csv')
    return render_template('test.html', test_message='Job Done?')


@app.route('/survey_extractor')
def survey_extract_page():
    return render_template('survey_extractor.html')


@app.route('/img_extract', methods=['GET', 'POST'])
def extraction():
    extract_files()
    return redirect(url_for('download_surveys_page'))

@app.route('/download_surveys', methods=['GET', 'POST'])
def download_surveys_page():
    return render_template('download_surveys.html')


@app.route('/download_surveys/downloading', methods=['GET', 'POST'])
def download_surveys():
    # download individual survey image files
    if request.method == 'GET':
        # Can't delete the file says process is still ongoing.
        # @after_this_request
        # def move_zip(response):
            # time.sleep(2)
            # return os.remove('SurVision/survey_files.zip', app.config['DOWNLOAD_FOLDER'])

        # Clean 'Upload' and 'Download' Folder
        clean_directory(app.config['UPLOAD_FOLDER'])
        clean_directory(app.config['DOWNLOAD_FOLDER'])

        return send_file('survey_files.zip', as_attachment=True)

@app.route('/setupkey', methods=['GET', 'POST'])
def setupkey_page():
    colours = ['Red', 'Blue', 'Black', 'Orange']
    return render_template('setupkey.html', colours=colours)

@app.route('/tabulation', methods=['GET', 'POST'])
def tabulation_page():
    # Get survey keys from the survey key folder ('.csv')
    # working on a sqlalchemy db to hold keys information
    survey_keys = [x for x in os.listdir(app.config['SURVEY_KEYS_FOLDER']) if x.endswith('.csv')]
    return render_template('tabulation.html', survey_keys=survey_keys)

@app.route('/tabulate_surveys', methods=['GET', 'POST'])
def tabulate_surveys():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        surveys = request.files.getlist('files[]')
        survey_key_name = request.form.get('survey_key')
        print(survey_key_name)
        try:
            # Clean directory before populating
            clean_directory(app.config['UPLOAD_FOLDER'])

            # fix this to save all incoming files.
            for survey in surveys:
                survey.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(survey.filename)))
        except:
            flash(f'There was a problem with the files you uploaded. Please make sure you are uploading .png files that were extracted from the extractor.', category='danger')

        # Execute survey scanning here.
        scn.execute_survey_scanning(survey_key_name)

    return render_template('tabulate_surveys.html',
                           survey_directory=os.listdir(app.config['UPLOAD_FOLDER']),
                           src=os.path.join(app.config['UPLOAD_FOLDER']))
