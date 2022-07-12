from program import app
from flask import render_template, redirect, url_for, flash, request, send_file, after_this_request
from werkzeug.utils import secure_filename
import os
from pdf2image import convert_from_path
from shutil import make_archive, move
import time


@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')


@app.route('/survey_extractor')
def survey_extract_page():
    return render_template('survey_extractor.html')


@app.route('/img_extract', methods=['GET', 'POST'])
def extraction():
    if request.method == 'POST':
        f = request.files['file']
        # clear old files
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))

        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        surveyPDFpath = "".join((app.config['UPLOAD_FOLDER'], f.filename))

        # Extract script here

        from extract import getSurveyFiles, rename_files
        conversion_path = app.config['DOWNLOAD_FOLDER']
        # clear old files
        for file in os.listdir(conversion_path):
            os.remove(os.path.join(conversion_path, file))

        images = convert_from_path(surveyPDFpath, dpi=150, fmt='png', grayscale=True,
                                   poppler_path=r"C:\Users\gregj\PycharmProjects\SurVision_Flask\venv\Lib\site-packages\poppler-21.11.0\Library\bin")
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

        return send_file('survey_files.zip', as_attachment=True)

@app.route('/tabulation', methods=['GET', 'POST'])
def tabulation_page():
    return render_template('tabulation.html')


"""
@app.route('/register', methods=['GET', 'POST'])
def register_page():
    form = RegisterForm()
    if form.validate_on_submit():
        user_to_create = User(username=form.username.data,
                              email_address=form.email_address.data,
                              password=form.password1.data)
        db.session.add(user_to_create)
        db.session.commit()
        login_user(user_to_create)
        flash(f'Account created Successfully! You are logged in as {user_to_create.username}', category='success')
        return redirect(url_for('market_page'))
    if form.errors != {}: #If there are not errors from the validations
        for err_msg in form.errors.values():
            flash(f'There was an error with creating a user: {err_msg}', category='danger')

    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    form = LoginForm()
    if form.validate_on_submit():
        attempted_user = User.query.filter_by(username=form.username.data).first()
        if attempted_user and attempted_user.check_password_correction(
                attempted_password=form.password.data
        ):
            login_user(attempted_user)
            flash(f'Success! You are logged in as {attempted_user.username}', category='success')
            return redirect(url_for('market_page'))
        else:
            flash('Username and password are not matched! Please try again!', category='danger')

    return render_template('login.html', form=form)

@app.route('/logout')
def logout_page():
    logout_user()
    flash("You have been logged out!", category='info')
    return redirect(url_for("home_page"))
    
"""
