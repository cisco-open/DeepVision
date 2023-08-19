#!/usr/bin/env python3
from flask import Flask, request, render_template
import sys
sys.path.append('..')
from Ethosight import Ethosight
import os

app = Flask(__name__)

# Instantiate Ethosight class
ethosight_dir = "../"
ethosight = Ethosight(ethosight_dir=ethosight_dir)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', msg='No file selected')
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', msg='No file selected')

        if file:
            image_path = os.path.join('./static', file.filename)
            directory = os.path.dirname(image_path)

            # check if directory exists
            if not os.path.exists(directory):
                # if not, create the directory
                os.makedirs(directory)

            # now you can save the file
            file.save(image_path)
            
            # load embeddings
            embeddings_filename =  model_dir = os.path.join(ethosight_dir, "general.embeddings")
            embeddings = ethosight.load_embeddings_from_disk(embeddings_filename)

            # compute and print affinity scores
            affinity_scores = ethosight.compute_affinity_scores(embeddings, image_path)

            # transform your data to list of tuples for easier rendering
            affinity_scores = [(label, score) for label, score in zip(affinity_scores['labels'], affinity_scores['scores'])]

            return render_template('index.html',
                                   msg='Successfully processed',
                                   affinity_scores=affinity_scores,
                                   img_file=file.filename)

    elif request.method == 'GET':
        # render upload page
        return render_template('index.html')

if __name__ == '__main__':
    app.run()
