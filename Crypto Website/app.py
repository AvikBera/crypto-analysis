import os
import uuid
import shutil
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename

# Import the functions from our new predictor module
# This is the correct "routing"
from model.predictor import run_prediction_pipeline, is_model_ready

# --- Constants ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_super_secret_key_12345'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main upload page."""
    # Call the helper function from our imported module
    model_ready = is_model_ready()
    return render_template('index.html', model_ready=model_ready)


@app.route('/predict', methods=['POST'])
def predict():
    """Handles file uploads and shows the results."""
    
    if 'csv_files' not in request.files:
        flash('No file part in request.', 'danger')
        return redirect(url_for('index'))
        
    files = request.files.getlist('csv_files')
    
    if not files or files[0].filename == '':
        flash('No files selected for uploading.', 'warning')
        return redirect(url_for('index'))

    session_id = str(uuid.uuid4())
    session_upload_path = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(session_upload_path)
    
    file_paths = []
    saved_filenames = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(session_upload_path, filename)
            file.save(file_path)
            file_paths.append(file_path)
            saved_filenames.append(filename)

    if not file_paths:
        shutil.rmtree(session_upload_path)
        flash('No valid .csv files were uploaded.', 'danger')
        return redirect(url_for('index'))

    # Call the prediction pipeline from our imported module
    metrics, plots, error = run_prediction_pipeline(file_paths)
    
    try:
        shutil.rmtree(session_upload_path)
    except Exception as e:
        print(f"Warning: Could not delete temp folder {session_upload_path}: {e}")
    
    return render_template(
        'results.html', 
        metrics=metrics, 
        plots=plots, 
        error=error,
        filenames=saved_filenames
    )

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)