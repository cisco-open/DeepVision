
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
from .EthosightApp import EthosightApp

@click.group()
def cli():
    pass

@cli.command()
@click.argument('app_dir')
@click.argument('config_file')
def create_app(app_dir, config_file):
    """Creates an EthosightApp instance."""
    result = EthosightApp.create_app(app_dir, config_file)
    
    if isinstance(result, str):  # error message returned
        click.echo(result)
    else:
        click.echo(f"Created EthosightApp at directory: {app_dir}")

@cli.command()
@click.argument('app_dir')
def delete_app(app_dir):
    """Deletes an EthosightApp instance."""
    result = EthosightApp.delete_app(app_dir)

    if isinstance(result, str):  # error message returned
        click.echo(result)
    else:
        click.echo(f"Deleted EthosightApp at directory: {app_dir}")

@cli.command()
@click.argument('app_dir')
@click.argument('top_n_labels')
def map(app_dir, top_n_labels):
    """Maps top-n labels to a top-1 label."""
    app = EthosightApp(app_dir)
    result = app.mapping(top_n_labels)
    click.echo(f"Top-1 label: {result}")

@cli.command()
@click.argument('app_dir')
def benchmark(app_dir):
    """Computes accuracy on a directory of images."""
    app = EthosightApp(app_dir)
    top1_acc, topn_acc = app.benchmark()
    click.echo(f"Top-1 accuracy: {top1_acc}, Top-N accuracy: {topn_acc}")

@cli.command()
@click.argument('app_dir')
@click.argument('failure_cases')
def learn(app_dir, failure_cases):
    """Runs the iterative learning loop on a set of failure cases."""
    app = EthosightApp(app_dir)
    result = app.iterative_learning_loop(failure_cases)
    click.echo(f"Learning result: {result}")

@cli.command()
@click.argument('app_dir')
def get_failures(app_dir):
    """Gets the failure cases from the benchmarking."""
    app = EthosightApp(app_dir)
    result = app.get_failure_cases()
    click.echo(f"Failure cases: {result}")

@cli.command()
@click.argument('app_dir')
def optimize(app_dir):
    """Optimize the EthosightApp."""
    app = EthosightApp(app_dir)
    app.optimize()
    click.echo(f"Built the EthosightApp at directory: {app_dir}.")

@cli.command()
@click.argument('app_dir')
@click.argument('image')
def run(app_dir, image):
    """Runs the EthosightApp on a single image."""
    app = EthosightApp(app_dir)
    result = app.run(image)
    click.echo(f"Analysis result: {result}")

@cli.command()
@click.argument('app_dir')
@click.argument('video_gt_csv_filename')
def benchmark_video(app_dir, video_gt_csv_filename):
    """Runs video benchmarking on a video."""
    app = EthosightApp(app_dir)
    app.benchmark_videos_from_csv(video_gt_csv_filename)

@cli.command()
@click.argument('app_dir')
@click.argument('json_file_path')
def rank_affinities(app_dir, json_file_path):
    app = EthosightApp(app_dir)
    app.rank_affinities(json_file_path)

@cli.command()
@click.argument('app_dir')
@click.option('--phase2_groundtruth_csv', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
              help='Path to the Phase 2 Ground Truth CSV file.')
def phase2videobenchmarks(app_dir, phase2_groundtruth_csv):
    """runs benchmarks on all of the affinity score json files contained in the csv file. these are produced by phase1 Ethosight processing of video datasets"""
   # This is where you will use the phase2_groundtruth_csv, as per your application needs
    app = EthosightApp(app_dir)
    # The benchmarking mechanism might differ based on phase2_groundtruth_csv, so ensure to use it properly in your application
    app.phase2videobenchmarks(phase2_groundtruth_csv=phase2_groundtruth_csv)

@cli.command()
@click.argument('app_dir')
@click.argument('labels', nargs=-1)  # This will allow multiple label inputs
def add_labels(app_dir, labels):
    """Adds new labels to the EthosightApp."""
    app = EthosightApp(app_dir)

    # This assumes that EthosightApp has a method called `add_labels` that accepts a list of labels
    result = app.add_labels(list(labels))

    if isinstance(result, str):  # error message returned
        click.echo(result)
    else:
        click.echo(f"Added labels to EthosightApp at directory: {app_dir}")

if __name__ == "__main__":
    cli()