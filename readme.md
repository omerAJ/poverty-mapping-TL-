# Poverty Mapping Using Transfer Learning

## Project Aim
The project aims at exploring different techniques for developing a fine grained (high-res) poverty map for Pakistan. 
The main challenge is the lack of spatially distributed high resolution ground truth labels of poverty. Most available data is at a much coarser resolution (town level/district level). 
Satellite image at good resolution (spatial and temporal) is freely available for the entire world. Thus we have the aim to develop a model from satellite imagery to poverty labels. i.e., 
$f_{\theta}(sat\\_img) = poverty\\_label$




## Project Overview
Main aim of this repo is to implement the approach described in [Transfer Learning from Deep Features for Remote Sensing and Poverty Mapping [1]], 
utilizing transfer learning to generate poverty maps from night-time light (NTL) imagery. 
The problem is that satellite imagery is too unstructured and not enough labeled poverty data is available to train deep learning models to extract any viable information from them. Meanwhile abundant labels for night time lights (NTL) are available (https://www.earthdata.nasa.gov/learn/backgrounders/nighttime-lights). Thus, the idea is to first train a deep learning model on the task of NTL prediction from satellite imagery, the trained model can then be finetuned on the limited available poverty labels. 

The project is divided into two main parts: data collection and preprocessing (QGIS and python), and training (pretraining and transfer learning) and evaluating a deep learning model.

## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Data Collection](#data-collection)
  - [Training the Model](#training-the-model)
- [Project Structure](#project-structure)
- [Limitations and Future Work](#limitations-and-future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Getting Started

### Prerequisites
- python
- QGIS

### Installation
1. Clone the repository:
git clone https://github.com/omerAJ/poverty-mapping-TL-.git
cd poverty-mapping

2. Install required Python libraries:
pip install -r requirements.txt


## Usage

### Data Collection
This section details the steps to collect and preprocess the data needed for the project.
1. **Downloading Night-Time Light Imagery:**
- NTL Imagery can be downloaded from google earth engine using the script provided in Data/scripts/NTL_collection.txt

2. **Downloading Satellite Imagery:**
- Download satellite imagery for the areas corresponding to NTL data. 

3. **Join satellite imagery and NTL imagery to generate NTL labels for pretraining phase:**
- Use QGIS to create a grid on NTL. And use the grid to cut the sattelite imagery .tif file to corresponding grid and use the labels from the NTL grid.
- Can use the code in Data/scripts/clip.ipynb  

### Data Available
Data downloaded for the project is availble in city3 in D://omer/poverty_mapping_data
1. Poverty survey Sialkot {D://omer/poverty_mapping_data/poverty_survey}
2. NTL for Sialkot {D://omer/poverty_mapping_data/}
3. Satellite imagery for Sialkot {D://omer/poverty_mapping_data/sat_imagery Sialkot}
4. Satellite imagery for Sialkot with the corresponding NTL value as its label {D://omer/poverty_mapping_data/clipped_data} (the matching of NTL label and sat imagery was done using QGIS, and python)
5. Poverty data of districts in Sindh as used in the paper [High-resolution rural poverty mapping in Pakistan with ensemble deep learning [2]] data is in .tif format can be visualized in QGIS. For more details refer to the paper, it also provides additional scripts to process the data. {D://omer/poverty_mapping_data/data_memon}


### Training the Model
Two stage training process pretraining, and transfer learning
1. Pretraining code is in train.ipynb
2. Transfer learning code is in train_poverty.ipynb


## Limitations and Future Work
- **Limitations:**
  - The major limitation is still the availability of high resolution poverty ground truth. Even transfer learning requires few high quality labels for transfering the model from NTL to Poverty. The Sialkot survey contains data in big ranges and it is insufficient to devise a viable poverty index from this survey. Thus we were unable to transfer our model trained on NTL to poverty. 
  - Some (talha bhai) say that poverty can't be mapped from satellite imagery at a fine resolution because satellite imagery is just not informative enough for this task, and there are way too many intricacies. 
  - The paper [High-resolution rural poverty mapping in Pakistan with ensemble deep learning https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0283938] uses very poor quality satellite imagery for training (can be visualized on qgis from the data folder). Further the ground truth in the sindh survey is also with added noise (the noise is greater than the acclaimed high resolution of the generated poverty maps).

- **Future Work:**
  - Future work should focus on developing a high reoslution poverty label first e.g., use spatially referenced telecom data to devise poverty index. Or some other way. Defining the dependent variable (y) is the toughest part of this project. 
  - Unsupervised or self supervised techniques can be tried but are likely to fail, as poverty is a very inherent/hidden (even if present) information in satellite imagery which the un/self -supervised techniques are very unlikely to uncover. You will need a strong supervision to extract poverty information from satellite imagery.

## Contributing
Full credits to the two mentioned papers.

[1] Xie M, Jean N, Burke M, Lobell D, Ermon S. Transfer Learning from Deep Features for Remote Sensing and Poverty Mapping. arXiv:1510.00098 [cs.CV]. https://arxiv.org/abs/1510.00098

[2] gyemang FSK, Memon R, Wolf LJ, Fox S (2023) High-resolution rural poverty mapping in Pakistan with ensemble deep learning. PLoS ONE 18(4): e0283938. https://doi.org/10.1371/journal.pone.0283938


## Acknowledgements
- This project was done at Centre for Urban Informatics, Technology, and Policy (CITY @ LUMS)

