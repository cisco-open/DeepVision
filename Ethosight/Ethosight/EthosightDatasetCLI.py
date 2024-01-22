#!/usr/bin/env python3
import os
import csv
import glob
import click
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
from difflib import SequenceMatcher

@click.group()
def cli():
    pass

@click.command()
@click.option('--dir', default='./', help='Directory with .txt files containing URLs')
def imagecounts(dir):
    """
    Counts the number of lines in the .txt files located in the provided directory and
    prints the count for each file, as well as a total count across all files.
    """

    # Make sure path ends with '/'
    if not dir.endswith('/'):
        dir += '/'

    total_count = 0

    for file in os.listdir(dir):
        if file.endswith('.txt'):
            with open(dir + file, 'r') as f:
                lines = f.readlines()
                count = len(lines)
                print(f'{file}: {count} links')
                total_count += count

    print(f'Total count: {total_count} links')

def log_failed_url(file: str, url: str):
    """Helper function to log failed URLs and their source files."""

    with open('failed_urls.txt', 'a') as f:
        f.write(f'Failed URL from {file}: {url}\n')

def download_images(file):
    """Helper function to download images from the URLs in a given text file."""

    base_file_name = Path(file).stem
    output_dir = Path(file).parent / base_file_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(file, 'r') as f:
        urls = [line.strip() for line in f]

    with click.progressbar(urls, label="Processing URLs") as bar:
        for url in bar:
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Raise an HTTPError if the response was unsuccessful.
            except (requests.RequestException, ValueError):
                click.echo(f'Failed to process URL {url}')
                log_failed_url(file, url)  # Log the failed URL and its source file
                continue

            # Use the last path component as the filename
            filename = urlparse(url).path.split('/')[-1]
            filename = base_file_name + "_" + filename

            filepath = output_dir / filename

            with open(filepath, 'wb') as out_file:
                out_file.write(response.content)

@click.command()
@click.argument('file', type=click.Path(exists=True))
def imagedownload(file):
    download_images(file)


@click.command()
@click.argument('dir', default='.', type=click.Path(exists=True))
def download_all(dir):
    txt_files = glob.glob(os.path.join(dir, '*.txt'))

    for txt_file in txt_files:
        download_images(txt_file)

def list_video_affinity_files(directory):
    affinity_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                affinity_files.append(os.path.relpath(os.path.join(root, file)))
    return affinity_files

def read_ground_truth_labels(filename):
    """Read the ground truth labels from the file."""
    with open(filename, 'r') as f:
        return [line.strip().lower() for line in f.readlines()]

def best_matching_label(filename, labels):
    """Find the best matching label using case insensitive matching. If multiple labels match equally, the first one is returned."""
    filename = filename.lower()
    best_match = None
    best_score = 0
    for label in labels:
        score = SequenceMatcher(None, filename, label).ratio()
        if score > best_score:
            best_score = score
            best_match = label
    return best_match

def write_csv(output_csv, labels_file, affinities_directory):
    affinity_files = list_video_affinity_files(affinities_directory)
    ground_truth_labels = read_ground_truth_labels(labels_file)
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['affinity_file', 'groundtruth_label'])
        for affinity_file in affinity_files:
            label = best_matching_label(affinity_file, ground_truth_labels)
            writer.writerow([affinity_file, label])

@click.command()
@click.argument('output_csv')
@click.argument('labels_file', type=click.Path(exists=True))
@click.argument('affinities_directory', type=click.Path(exists=True))
def generate_csv_for_video_affinities(output_csv, labels_file, affinities_directory):
    write_csv(output_csv, labels_file, affinities_directory)

cli.add_command(imagecounts)
cli.add_command(imagedownload)
cli.add_command(download_all)
cli.add_command(generate_csv_for_video_affinities)


if __name__ == '__main__':
    cli()
