#!/usr/bin/env python3
import click
import os
from Ethosight import Ethosight, ChatGPTReasoner
#from ChatGPTReasoner import ChatGPTReasoner
#from NARSReasoner import NARSReasoner
#from NARSGPTReasoner import NARSGPTReasoner
#from llama_index_reasoner import LlamaIndexReasoner
#from Ethosight import Ethosight

USE_CASE_DEFAULT = 'child in danger'
LABEL_AFFINITY_SCORES_DEFAULT = ''
PROMPT_TYPE_DEFAULT = 'blank_slate'

@click.group()
def cli():
    pass

@cli.command()
@click.argument('image_filename')
@click.option('--background-knowledge', '-bg', default="", help='The new-line separate background knowledge of the reasoner, whereby each line can be English or Narsese. Default is "".')
@click.option('--summary-file', '-s', default=None, help="The file containing the summary sentence that was obtained (mandatory)")
@click.option('--questions', '-q', default="what happened, please give a lengthy answer mentioning everything you know?", help='The questions to be answered, new-line-separated<F12>.', required=True)
@click.option('--outfile', '-o', default='summary.txt', help='Output file name to write summary sentence. Default: summary.txt.', required=False)
@click.option('--debug', '-d', is_flag=True, help='Enable debug mode', required=False)
def ask(image_filename: str, background_knowledge: str, summary_file: str, questions: str, outfile: str, debug=False):
    ethosight = Ethosight()
    reasoner = NARSGPTReasoner()
    for x in background_knowledge.split("\n"):
        x = x.strip()
        if x != "":
            reasoner.reason(x)
    with open(summary_file,"r") as f:
        summary = f.read()
    reasoner.reason(summary)
    question_answer=""
    for x in questions.split("\n"):
        question = x
        x = x.strip().replace("?", "")
        if x != "":
            labels = reasoner.labels_from_sentence(question)
            if labels:
                newembeddings = ethosight.compute_label_embeddings(labels)
                affinity_scores = ethosight.compute_affinity_scores(newembeddings, image_filename)
                summary = reasoner.summarize(affinity_scores)
                reasoner.reason(summary)
            answer = reasoner.reason(question)["GPT_Answer"]
            if debug:
                click.echo(f"Summary for question '{question}':{ summary}")
            click.echo(f"Answer to question '{question}': {answer}")
            question_answer += "\n" + str((question, answer))
    if outfile:
        with open(outfile, "w") as f:
            f.write(question_answer)
        click.echo(f"Questions and answers written to file {outfile}.")

@cli.command()
#@click.option('--use-case', '-u', default=USE_CASE_DEFAULT, help='The use case for the reasoner. Default is "{USE_CASE_DEFAULT}".')
@click.option('--label-affinity-scores', '-l', default=LABEL_AFFINITY_SCORES_DEFAULT, help='The affinity scores for labels. Can be a string of labels or a path to a file. Default is an empty string.', required=False)
#@click.option('--prompt-type', '-p', default=PROMPT_TYPE_DEFAULT, help='The type of prompt for the reasoner. Can be "blank_slate" or "iterative". Default is "{PROMPT_TYPE_DEFAULT}".')
@click.option('--outfile', '-o', default='summary.txt', help='Output file name to write summary sentence. Default: summary.txt.', required=False)
@click.option('--debug', '-d', is_flag=True, help='Enable debug mode', required=False)
def summarize(label_affinity_scores: str, outfile: str, debug=False):
    """
    This command takes in a set of labels with their respective affinity scores, and generates a new set of labels based on the existing ones. 

    Example usage:
        ./EthosightCLI.py reason --filename path/to/labels/file  -> Generates new labels and saves them to 'path/to/labels/file'.
        ./EthosightCLI.py reason -> Prints the new labels to the console.
    """

    reasoner = NARSGPTReasoner()

    if os.path.isfile(label_affinity_scores):
        with open(label_affinity_scores, 'r') as file:
            label_affinity_scores = file.read()

    label_affinity_scores = [line.split(',') for line in label_affinity_scores.strip().split('\n') if ',' in line and line.strip()]
    labels = [x[0] for x in label_affinity_scores]
    scores = [float(x[1]) for x in label_affinity_scores] 
    label_affinities = {"labels" : labels, "scores" : scores}
    # Create a summary sentence from label affinities
    summary = reasoner.summarize(label_affinities)
    click.echo("Summary: " + summary)
    if outfile:
        with open(outfile, "w") as f:
            f.write(summary)
        click.echo(f"Summary written to file {outfile}.")

@cli.command()
@click.option('--use-case', '-u', default=USE_CASE_DEFAULT, help='The use case for the reasoner. Default is "{USE_CASE_DEFAULT}".')
@click.option('--label-affinity-scores', '-l', default=LABEL_AFFINITY_SCORES_DEFAULT, help='The affinity scores for labels. Can be a string of labels or a path to a file. Default is an empty string.', required=False)
@click.option('--prompt-type', '-p', default=PROMPT_TYPE_DEFAULT, help='The type of prompt for the reasoner. Can be "blank_slate" or "iterative". Default is "{PROMPT_TYPE_DEFAULT}".')
@click.option('--outfile', '-o', default='reasoner.labels', help='Output file name to write new labels. Default: reasoner.labels.', required=False)
@click.option('--debug', '-d', is_flag=True, help='Enable debug mode', required=False)
def reason(use_case: str, label_affinity_scores: str, prompt_type: str, outfile: str, debug=False):
    """
    This command takes in a set of labels with their respective affinity scores, and generates a new set of labels based on the existing ones. 

    Example usage:
        ./EthosightCLI.py reason --filename path/to/labels/file  -> Generates new labels and saves them to 'path/to/labels/file'.
        ./EthosightCLI.py reason -> Prints the new labels to the console.
    """

    reasoner = ChatGPTReasoner(use_case)
    #reasoner = LlamaIndexReasoner(use_case)
    #reasoner = NARSReasoner(use_case)

    if os.path.isfile(label_affinity_scores):
        with open(label_affinity_scores, 'r') as file:
            label_affinity_scores = file.read()

    label_affinity_scores = dict(line.split(',') for line in label_affinity_scores.strip().split('\n') if ',' in line and line.strip())

    if debug:
        print(f"Label affinity scores: {label_affinity_scores}")
        exit()

    new_labels = reasoner.reason(label_affinity_scores, prompt_type)

    if outfile:
        with open(outfile, 'w') as file:
            file.write('\n'.join(new_labels))
        click.echo(f"New labels written to file {outfile}.")

    click.echo("New labels:")
    for i, label in enumerate(new_labels, start=1):
        click.echo(f"{i}. >>{label}<<")

@click.command()
@click.argument('filename', type=click.Path(exists=True))
def embed(filename):
    """
    Compute the label embeddings from a file of labels.
    
    This command reads a list of labels from a file, computes their embeddings
    using the Ethosight model, and saves these embeddings to an embeddings file.

    FILENAME: The name of the file with the labels. Each line in the file should contain one label.
              The embeddings will be saved to a file with the same name, but with '.embeddings' as the extension.

    Example usage:

        ./EthosightCLI.py embed path/to/labels/file

    This will compute the embeddings for the labels in 'path/to/labels/file' and save them to 'path/to/labels/file.embeddings'.
    """
    ethosight = Ethosight()
    ethosight.embed_labels_from_file(filename)

    filename_without_extension, _ = os.path.splitext(filename)
    click.echo(f"Embeddings written to file {filename_without_extension}.embeddings.")
cli.add_command(embed)

# Define affinities method
@click.command(help='Compute affinity scores for an image with respect to the embeddings stored in a file. Save these scores to another file. The default filename for saving is the base name of the image file with the extension ".affinities".')
@click.argument('image_filename')
@click.argument('embeddings_filename')
@click.option('--output_filename', default=None, help='Optional: Path to the output file.')
def affinities(image_filename, embeddings_filename, output_filename):
    # Check if image file exists
    if not os.path.isfile(image_filename):
        raise click.BadParameter(f"Image file does not exist: {image_filename}")

    # Check if embeddings file exists
    if not os.path.isfile(embeddings_filename):
        raise click.BadParameter(f"Embeddings file does not exist: {embeddings_filename}")

    ethosight = Ethosight()
    label_to_embeddings = ethosight.load_embeddings_from_disk(embeddings_filename)
    affinities = ethosight.compute_affinity_scores(label_to_embeddings, image_filename)

    if output_filename is None:
        base_filename, _ = os.path.splitext(image_filename)
        output_filename = f"{base_filename}.affinities"

    # Strip directory information from output_filename
    # output_filename = os.path.basename(output_filename)

    with open(output_filename, 'w') as f:
        # Write header line
        #f.write("label,score\n")
        # Write affinity scores
        for label, score in zip(affinities['labels'], affinities['scores']):
            f.write(f"{label},{score}\n")

    click.echo(f"Affinity scores written to file {output_filename}.")
cli.add_command(affinities)

@click.command(help='Initiate the manual learning loop. This process will prompt the user for new labels and present the top 10 affinity scores for each iteration.')
@click.argument('image_file', type=click.Path(exists=True))
@click.argument('use_case_prompt')  
@click.argument('normalize_fn', type=click.Choice(['softmax', 'sigmoid', 'linear']))
def manual_learning_loop(image_file, use_case_prompt, normalize_fn):
    ethosight = Ethosight()
    affinities_filename = ethosight.manual_learning_loop(image_file, use_case_prompt, normalize_fn)
    click.echo(f"The final affinity scores are saved in {affinities_filename}.")

cli.add_command(manual_learning_loop)

if __name__ == '__main__':
    cli()
