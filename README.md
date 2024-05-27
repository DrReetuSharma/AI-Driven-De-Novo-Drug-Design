# AI-Driven Drug Design
#### Generating Novel Molecules with Desired Properties

## Overview:
This repository contains code and resources for utilizing AI-driven methods to design novel drug molecules with desired properties. Leveraging Generative Adversarial Networks (GANs) combined with reinforcement learning (RL) techniques, the goal is to generate diverse and high-quality drug candidates optimized for efficacy, selectivity, and pharmacokinetic profiles.

## Features:

Implementation of GAN-based models for generating molecular structures.
Integration of reinforcement learning algorithms to optimize molecular properties.
Preprocessing scripts for data collection and preparation.
Evaluation tools for assessing generated molecules' drug-likeness and bioactivity predictions.
Examples and tutorials demonstrating the usage of AI-driven methods for drug design.

## Dependencies:
Python 3.x
TensorFlow or PyTorch (for GAN implementation)
OpenAI Gym (for RL algorithms)
RDKit (for molecular manipulation and analysis)
Pandas, NumPy, Matplotlib (for data processing and visualization)
License:
This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments:

This work is inspired by the advancements in AI-driven drug discovery and the contributions of researchers in the field.
Special thanks to the developers of open-source libraries and datasets used in this project.
References:

https://aspire10x.com/data-solutions/


## Structure

- Root
  |- README.md
  |- LICENSE
  |- requirements.txt
  |- src/
  |   |- train_gan.py
  |   |- train_rl.py
  |   |- evaluate_molecules.py
  |- data/
  |   |- dataset.csv
         preprocessed_dataset.csv
        features_array.npy ( result of train_molgan.py)
        adj_array.npy   ( result of train_molgan.py)
  
  |- docs/
  |   |- user_manual.md
  |   |- api_documentation.md
  |- examples/
  |   |- example_notebook.ipynb
  |- tests/
  |   |- test_gan.py
  |   |- test_rl.py
  |- scripts/
  |   |- setup.sh
  |   |- preprocess_data.py
  |- contrib/
      |- contribution_guidelines.md


preprocess_data.py input data/preprocessed_dataset.csv  output:adj_array.npy, feature_array.npy ( smiles to graph)
train_molgan.py   input:.npy  output:  generated_molecules_df.to_csv('data/generated_molecules.csv', index=False)
src/train_rl.py input: models/generator_final.pth  models/discriminator_final.pth


 

## Contact/correspondance:
For any inquiries or feedback, please contact sharmar@aspire10x.com.
https://aspire10x.com/data-solutions/
