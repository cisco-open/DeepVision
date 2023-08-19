import csv
import os
import click
import base64
import glob
import requests
import sys
from urllib.parse import urlparse
from pathlib import Path

# processing files with image encoded in url
def process_url_data(url, base_file_name, output_dir):
    url_data = url.split(',')[1]
    image_data = base64.b64decode(url_data)
    filename = base_file_name

    # appending number to filename to avoid overwriting
    filename = append_number_to_file(filename, output_dir)

    filepath = output_dir / filename
    with open(filepath, "wb") as f:
        f.write(image_data)

# function to append number to filename if file already exists
def append_number_to_file(filename, output_dir):
    basename, extension = os.path.splitext(filename)
    file_in_dir = os.path.join(output_dir, basename)
    file_num = 1
    basename1 = basename
    while os.path.exists(file_in_dir) or os.path.exists(file_in_dir + extension) or os.path.exists(file_in_dir + '.jpg'):
        basename1 = basename + str(file_num)
        file_in_dir = os.path.join(output_dir, basename1)
        file_num += 1
    if extension == '':
        extension = '.jpg'
    return basename1 + extension


def download_images(file):

    base_file_name = Path(file).stem
    output_dir = Path(file).parent / base_file_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(file, 'r') as f:
        urls = [line.strip() for line in f]

    with click.progressbar(urls, label="Processing URLs") as bar:
        for url in bar:
            try:
                if url.startswith("data:image"):
                    process_url_data(url, base_file_name, output_dir)
                else:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    filename = urlparse(url).path.split('/')[-1]
                    filename = base_file_name + "_" + filename

                    # append an integer if folder contains files with same names
                    filename = append_number_to_file(filename, output_dir)

                    # write to file
                    filepath = output_dir / filename
                    with open(filepath, 'wb') as out_file:
                        out_file.write(response.content)

            except (requests.RequestException, ValueError):
                click.echo(f'Failed to process URL {url}')
                log_failed_url(file, url)
                continue

           

def log_failed_url(file: str, url: str):
    with open('failed_urls.txt', 'a') as f:
        f.write(f'Failed URL from {file}: {url}\n')

def download_all(dir):
    txt_files = glob.glob(os.path.join(dir, '*.txt'))
    for txt_file in txt_files:
        download_images(txt_file)

# function to list all files in a directory and add to csv
def list_files(dir):
    file_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.txt'):
                continue
            file_list.append([file, root.split('/')[-1]])

    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Image', 'Label'])
        csv_writer.writerows(file_list)

if __name__ == '__main__':
    dir = sys.argv[1]
    csv_file_path = sys.argv[2]
    download_all(dir)
    list_files(dir)