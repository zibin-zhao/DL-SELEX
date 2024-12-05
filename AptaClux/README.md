# AptaClux Workflow

This repository contains scripts and files for the **AptaClux** model in the **DL-SELEX** project. 

AptaClux is a deep learning-based software aimed at providing the most straightforward and effective aptamer generation from HT-SELEX results (commonly HT-NGS data post-analysis). The underlying model is a conditional variational autoencoder (VAE) including both sequential and structural information (2D). Structural information has been proven to be key during aptamer-ligand interaction, whereas current available models often lack inclusion of this information due to calculation and implementation difficulties. The clustering methods chosen in AptaClux were compared and fine-tuned based on our steroid targets, which can also be fine-tuned for user-specific targets with downloading. The AptaClux-generated aptamers were experimentally validated with the current gold standard isothermal titration calorimetry (ITC) for reliability testing. The results can be found in the manuscript DL-SELEX.

The AptaClux web server provides the simplest and most user-friendly interface for any user to analyze their HT-SELEX NGS data with a deep learning model in one click. Users can input the NGS data as a FASTA datatype into the server, and two predicted aptamer sequences will be printed out or emailed to the user after model training and sampling.

## Quickstart with local build (Linux / macOS / Windows(subsystem)

1. Download the source_code in the current directory.

2. Ensure python version > 3.9 is installed locally the create a virtual environment.

```shell
% pip install virtualenv
% cd project_folder
% virtualenv venv
% source venv/bin/activate
```

3. Installed NUPACK following the link: https://docs.nupack.org/start/#maclinux-installation

4. Installed the required packages with
```shell
% pip install -r requirements.txt
```

5. Quick run with sample.fasta
```shell
% python run.py -i sample.fasta -o output.txt
```

You can also specifify options:
```shell
$ python run.py --help
usage: run.py [-h] -i INPUT [-o OUTPUT] [-nb NUM_DEVICES] [-max MAX_EPOCH] [-tp TEMPERATURE] [-ions IONS] [-oligos OLIGOS] [-seed SEED]

AptaClux Model Training and Sampling for aptamer candidates generation.

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to the input FASTA file.
  -o OUTPUT, --output OUTPUT
                        Path to the output text file for results.
  -nb NUM_DEVICES, --num_devices NUM_DEVICES
                        Number of GPUs (or CPUs if none) to use.
  -max MAX_EPOCH, --max_epoch MAX_EPOCH
                        Maximum number of epochs for training.
  -tp TEMPERATURE, --temperature TEMPERATURE
                        Temperature in degrees Celsius for the model.
  -ions IONS, --ions IONS
                        Ion concentration, default is sodium=0.147M.
  -oligos OLIGOS, --oligos OLIGOS
                        Oligonucleotide type, default is DNA.
  -seed SEED, --seed SEED
                        Random seed for reproducibility.
```

## Detailed model workflow

Other than source_code, the related more detailed code for model training and preprocessing can be also found in this directory.

## License

This project is licensed under the MIT License.

## Contact

For further questions, please contact zzhaobz@connect.ust.hk.
