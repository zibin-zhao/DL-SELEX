import subprocess
import os

def process_uploaded_file(file_path, form_data):
    result_file = 'output.txt'
    command = [
        "python3", "run.py", 
        "-i", file_path, 
        "-o", result_file,
        "-max", form_data.get("max_epoch", "2000"),
        "-tp", form_data.get("temperature", "25"),
        "-ions", form_data.get("ions", "0.147"),
        "-oligos", form_data.get("oligos", "dna"),
        "-seed", form_data.get("seed", "42")
    ]
    subprocess.run(command, check=True)

    result_file_path = os.path.abspath(result_file)

    # Read sequences from the result file to display on the webpage
    with open(result_file_path, 'r') as f:
        sequences = f.read()
    
    return result_file_path, sequences
