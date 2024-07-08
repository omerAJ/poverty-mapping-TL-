# Poverty Mapping Using Transfer Learning

## Project Aim
The project aims at exploring different techniques for developing a fine grained (high-res) poverty map for Pakistan. 
The main challenge is the lack of spatially distributed high resolution ground truth labels of poverty. Most available data is at a much coarser resolution (town level/district level). 
Satellite image at good resolution (spatial and temporal) is freely available for the entire world. Thus we have the aim to develop a model from satellite imagery to poverty labels. i.e., 
<img src="https://render.githubusercontent.com/render/math?math=f(\text{sat\_imagery}) = \text{poverty labels}">


## Project Overview
Main aim of this repo is to implement the approach described in [Transfer Learning from Deep Features for Remote Sensing and Poverty Mapping (https://arxiv.org/pdf/1510.00098)], 
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
- Instructions on how to download NTL imagery from QGIS.
- Specific settings or filters used during the download.

2. **Downloading Corresponding Labels:**
- Steps to download poverty labels from QGIS.
- How to align these labels with the NTL imagery data.

3. **Data Preprocessing:**
- Scripts used to preprocess the data.
- Example command to run the preprocessing script:
  ```
  python preprocess_data.py
  ```

### Training the Model
- Detailed steps on how to train the deep learning model.
- Example command to start training:

python train_model.py


## Project Structure
poverty-mapping/
│
├── data/ # Folder for raw and preprocessed data
├── models/ # Trained model files and model definitions
├── scripts/ # Scripts for data preprocessing and model training
├── requirements.txt # Python dependencies required
└── README.md # Project README file


## Limitations and Future Work
- **Limitations:**
  - Discuss any limitations encountered in the dataset, model, or performance.
  - Any issues with data quality or quantity.

- **Future Work:**
  - Suggestions for future improvements on the model or data.
  - Potential research questions or enhancements based on your insights.

## Contributing
Feel free to fork this project and submit your contributions via pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements
- Mention any organizations, data providers, or individuals who helped facilitate this project.

