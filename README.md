# Supplementary code for "Structure-enhanced Deep Learning Accelerates Aptamer Selection for Small Molecule Families like Steroids"

[![DOI](https://zenodo.org/badge/898798378.svg)](https://doi.org/10.5281/zenodo.14281817)

## Tested environment

* Ubuntu == 20.04
* python == 3.9
* pytorch == 1.5.0
* cuda == 12.1

## AptaVAE

AptaVAE is designed as the first part of the DL-SELEX approach, which generates the pre-defined guided initial library of the collected steroids. The detailed workflow of how the code works can be found inside the AptaVAE folder with the README file. 

## AptaClux

AptaClux as the second part of the DL-SELEX approach, can be accessed via our online web server at http://hsingapp.ust.hk. To deploy the model locally for own use, detailed procedure can be found in the README file under the AptaClux directory.

## Models comparison

We have also compared the performance of our AptaClux to current available deep learning models (i.e. AptaDiff, RaptGen), all comparison files have been deposited.
