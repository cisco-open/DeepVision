
# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
import click
import yaml
from Ethosight import EthosightMediaAnalyzer

@click.group()
def cli():
    pass

@click.command()
@click.argument('configfile', type=click.Path(exists=True))
def validateConfig(configfile):
    """
    Validate the configuration file
    CONFIGFILE is the path to the YAML configuration file.
    """
    # Validate the config file using EthosightMediaAnalyzer method
    try:
        em_analyzer = EthosightMediaAnalyzer(configfile)
        click.echo('Configuration is valid.')
    except ValueError as e:
        click.echo(f'Configuration validation error: {e}')

@click.command()
@click.argument('configfile', type=click.Path(exists=True))
@click.argument('imagepath', type=click.Path(exists=True))
def analyzeImage(configfile, imagepath):
    """
    Analyze the image
    CONFIGFILE is the path to the YAML configuration file.
    IMAGEPATH is the path to the image file.
    """
    # Instantiate the EthosightMediaAnalyzer with the config file
    em_analyzer = EthosightMediaAnalyzer(configfile)

    # Call the analyzeImage method with the image path
    output = em_analyzer.analyzeImage(imagepath)
    print(output)

@click.command()
@click.argument('configfile', type=click.Path(exists=True))
@click.argument('imagesfile', type=click.Path(exists=True))
def analyzeImages(configfile, imagesfile):
    """
    Analyze a list of images
    CONFIGFILE is the path to the YAML configuration file.
    IMAGESFILE is the path to a text file containing a list of image paths.
    """
    # Instantiate the EthosightMediaAnalyzer with the config file
    em_analyzer = EthosightMediaAnalyzer(configfile)
    
    # Open the images file
    with open(imagesfile, 'r') as f:
        # Loop through each line in the file (each line should be an image path)
        for line in f:
            # Strip out any leading/trailing whitespace (including newline characters)
            image_path = line.strip()
            
            # Check that the image path is not empty (which can happen if there are empty lines in the file)
            if image_path:
                # Call the analyzeImage method with the image path
                output = em_analyzer.analyzeImage(image_path)
                print(f"output for image {image_path}:")
                print(output)

@click.command()
@click.argument('configfile', type=click.Path(exists=True))
@click.argument('imagedir', type=click.Path(exists=True))
@click.argument('groundtruthfile', type=str)
@click.argument('topn', type=int)
def computeAccuracy(configfile, imagedir, groundtruthfile, topn):
    """
    Compute the accuracy of the model for a directory of images
    CONFIGFILE is the path to the YAML configuration file.
    IMAGEDIR is the path to the directory containing images.
    GROUNDTRUTHFILE is the path to the CSV file containing image labels.
    """
    # Instantiate the EthosightMediaAnalyzer with the config file
    em_analyzer = EthosightMediaAnalyzer(configfile)

    # Call the compute_accuracy_for_directory method with the image directory path and ground truth file
    top1_acc, top3_acc = em_analyzer.compute_accuracy_for_directory(imagedir, groundtruthfile, topn)
    print(f"Top 1 accuracy for images in {imagedir}: {top1_acc}")
    print(f"Top {topn} accuracy for images in {imagedir}: {top3_acc}")

cli.add_command(computeAccuracy)
cli.add_command(analyzeImages)
cli.add_command(validateConfig)
cli.add_command(analyzeImage)

if __name__ == '__main__':
    cli()
