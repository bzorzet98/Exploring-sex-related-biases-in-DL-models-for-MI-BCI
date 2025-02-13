# Exploring Sex-Related Biases in DL Models for MI-BCI

This repository contains the code and resources necessary to reproduce the experiments presented in the paper *"Exploring Sex-Related Biases in Deep Learning Models for Motor Imagery-Based Brain-Computer Interfaces (MI-BCI)"*. The provided scripts cover data preprocessing, model training, evaluation, and analysis.

## Environment Setup
To ensure reproducibility, the following Conda environments need to be created with their respective libraries.

### 1. Train Environment
This environment is used for model training and evaluation.
```bash
# Create the conda environment with Python 3.11.9
conda create --name train_environment python=3.11.9

# Install required libraries
conda install -c conda-forge scipy=1.14
conda install -c conda-forge scikit-learn

# Install PyTorch and torchvision (CPU version)
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 -c pytorch

# (Optional) Install PyTorch with CUDA support (GPU)
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Additional libraries
conda install -c conda-forge pandas
python -m pip install -U skorch
pip install einops
conda install -c conda-forge pyriemann
```

### 2. EEG Environment
This environment is required for EEG data preprocessing.
```bash
# Create the conda environment with Python 3.11.9
conda create --name eeg_environment python=3.11.9

# Install required libraries
conda install -c conda-forge scipy=1.14
conda install --channel conda-forge --strict-channel-priority mne
pip install pandas
pip install moabb==1.1.1
```

### 3. Analysis Environment
This environment is used for results analysis, plotting, and metric computation.
```bash
# Create the conda environment with Python 3.11.9
conda create --name analysis_environment python=3.11.9

# Install required libraries
conda install -c conda-forge seaborn
conda install pandas
pip install pingouin
pip install scikit-learn
```

## Repository Structure
- `preprocessing/`: Scripts for EEG data preprocessing.
- `training/`: Training scripts and model architectures.
- `analysis/`: Scripts for metrics computation and visualizations.
- `results/`: Output files and figures generated during experiments.

## Usage Instructions
1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/Exploring-sex-related-biases-in-DL-models-for-MI-BCI.git
    cd Exploring-sex-related-biases-in-DL-models-for-MI-BCI
    ```
2. Activate the appropriate environment depending on the task:
    ```bash
    conda activate train_environment   # For model training
    conda activate eeg_environment     # For EEG preprocessing
    conda activate analysis_environment # For results analysis
    ```
3. Run the corresponding scripts following the documentation within each folder.

## Citation
If you use this code or any derived results, please cite the paper as follows:
```bibtex
@article{YourName2024,
  title={Exploring Sex-Related Biases in Deep Learning Models for Motor Imagery-Based Brain-Computer Interfaces},
  author={Your Name and Co-authors},
  journal={Journal/Conference Name},
  year={2024},
  url={https://github.com/your-username/Exploring-sex-related-biases-in-DL-models-for-MI-BCI}
}
```

## License
This repository is licensed under [your chosen license, e.g., CC BY-NC-SA 4.0 or MIT], which allows usage with the condition of citation and prohibits modifications.

For more information, please refer to the paper or contact the corresponding author.


