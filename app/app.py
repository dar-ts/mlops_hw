import os
import shutil
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import json

# Import modelling scripts
import src.preprocessing as preprocessing
import src.scorer as scorer

ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_app():
    app = Flask(__name__)
    app.static_folder = 'static'

    def clean_directiry(directory):
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory)
    
    clean_directiry('input')
    clean_directiry('output')
    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        if request.method == 'POST':
            
            # Import file
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                
                # Store imported file locally
                new_filename = f'{filename.split(".")[0]}_{str(datetime.now())}.csv'
                save_location = os.path.join('input', new_filename)
                file.save(save_location)
                
                # Get input dataframe
                input_df = preprocessing.import_data(save_location)

                # Run preprocessing
                preprocessed_df = preprocessing.run_preproc(input_df)

                # Run scorer to get submission file for competition
                submission, model, probabilities = scorer.make_pred(preprocessed_df, save_location)
                output_filename = save_location.replace('input', 'output')
                submission.to_csv(output_filename.replace('.csv', '_predictions.csv'), index=False)

                #top 5 feature importances
                feature_names = preprocessed_df.columns
                top_features = scorer.get_top_features(model, feature_names)
                with open(output_filename.replace('.csv', '_features.json'), 'w') as f:
                    json.dump(top_features, f, ensure_ascii=False)
                
                #prediction distribution plot
                scorer.plot_prediction_distribution(probabilities, output_filename.replace('.csv', '_distribution.png'))

                return redirect(url_for('download'))

        return render_template('upload.html')

    @app.route('/download')
    def download():
        return render_template('download.html', files=os.listdir('output'))

    @app.route('/download/<filename>')
    def download_file(filename):
        return send_from_directory('output', filename)

    return app