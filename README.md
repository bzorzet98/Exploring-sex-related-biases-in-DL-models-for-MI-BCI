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
## Usage Instructions
1. Clone this repository:
    ```bash
    git clone https://github.com/bzorzet98/Exploring-sex-related-biases-in-DL-models-for-MI-BCI.git
    cd Exploring-sex-related-biases-in-DL-models-for-MI-BCI
    ```
2. Activate the appropriate environment depending on the task:
    ```bash
    conda activate train_environment   # For model training
    conda activate eeg_environment     # For EEG preprocessing
    conda activate analysis_environment # For results analysis
    ```
3. Run the corresponding scripts following the documentation within each folder.

## Script Descriptions
Due to the large volume of data, GitHub does not allow uploading all the result files. You can access all result files via this [link](your-link-here).

In the `EEG_DATASET` folder, there is a subfolder named `dataset_information`, which contains metadata information for each dataset. It is curated to standardize the metadata from both datasets into the same format.

### 1. `global_config.py`
This Python file contains predefined dictionaries and path configurations to manage results. By default, datasets and results are stored in the same repository folder. 
### 2. `init_paths.py`
This script configures the system paths by adding the appropriate source files depending on the environment.

### 3. `MNE_change_download_directory.py`
This script is necessary for redefining the paths from MNE, which is required for changing the default path when downloading datasets from MOABB.

Executed with the `eeg_environment`.

### 4. `download_datasets.py`
This script downloads the specified dataset using the `--dataset_to_download` flag. To download the datasets 'Cho2017' and 'Lee2019_MI', run this script as follows:
```bash
python download_datasets.py --dataset_to_download 'Cho2017'
python download_datasets.py --dataset_to_download 'Lee2019_MI'
```
Executed with the `eeg_environment`.

### 5. `metadata_simple_analysis.py`
This script processes the metadata of the dataset, selecting specific column names for analysis.
```bash
python metadata_simple_analysis.py --dataset_name 'Cho2017'
python metadata_simple_analysis.py --dataset_name 'Lee2019_MI'
```
Executed with the `analysis_environment`.

### 6. `preprocessing_datasets/preprocessing_datasets.py`
This script preprocesses the data from the databases using configuration files located in the `configs` folder. The preprocessing generates two `.npy` files per subject, session, and run (if applicable). These files are saved in `EEG_DATABASES/preprocessed_databases`. The preprocessed databases should be available via the provided link.

To run the script:
```bash
python preprocessing_datasets/preprocessing_datasets.py --script_config 'classical_preprocessing_Cho2017'
python preprocessing_datasets/preprocessing_datasets.py --script_config 'classical_preprocessing_Lee2019_MI'
```
Executed with the `eeg_environment`.

### 7. `train_LOSO_balanced_by_sex/train_LOSO_balanced_by_sex.py`
This script trains a model defined in the config file using a Leave-One-Subject-Out scheme, ensuring balanced training and validation sets. The config file contains various parameters to customize the training process.

The output files are saved in:
`results_path/experiment_name/model_to_train/database_name_database_session/timestamp`

If training is interrupted, you can resume by specifying the generated timestamp in the config file.

To run the script:
```bash
python train_LOSO_balanced_by_sex/train_LOSO_balanced_by_sex.py --script_config 'EEGNetv4_config_fairness_Cho2017'
python train_LOSO_balanced_by_sex/train_LOSO_balanced_by_sex.py --script_config 'EEGNetv4_config_fairness_Lee2019_MI_1'
python train_LOSO_balanced_by_sex/train_LOSO_balanced_by_sex.py --script_config 'EEGNetv4_config_fairness_Lee2019_MI_2'
```
Executed with the `train_environment`.

Training duration depends on several factors. Using a GPU, it takes approximately one week per database and session. For each subject, 100 models are trained with different training/validation splits and initialization seeds.

Each model folder contains the following files:
- `best_model.pt`: Parameters of the best model (lowest validation loss)
- `params_epoch_N.pt`: Model parameters at the first epoch, every 10 epochs, and the final epoch
- `test_outputs.csv`, `train_outputs.csv`, `val_outputs.csv`: Model predictions for each dataset split
- `info_data_training.json`: Dataset details (e.g., subjects in train/test/validation sets)
- `info_metrics_training.json`: Accuracy metrics for each dataset split
- `training_history.json`: Loss values per epoch, useful for plotting training curves

### 8. `train_LOSO_balanced_by_sex_CSP+LDA/train_LOSO_balanced_by_sex_CSP+LDA.py`
This script replicates the training process described above, but with a CSP+LDA model.

To run the script:
```bash
python train_LOSO_balanced_by_sex_CSP+LDA/train_LOSO_balanced_by_sex_CSP+LDA.py --script_config 'CSP+LDA_config_fairness_Cho2017'
python train_LOSO_balanced_by_sex_CSP+LDA/train_LOSO_balanced_by_sex_CSP+LDA.py --script_config 'CSP+LDA_config_fairness_Lee2019_MI_1'
python train_LOSO_balanced_by_sex_CSP+LDA/train_LOSO_balanced_by_sex_CSP+LDA.py --script_config 'CSP+LDA_config_fairness_Lee2019_MI_2'
```
Executed with the `eeg_environment`.

### 9. `analysis/compute_metrics_from_outputs.py`
This script calculates and aggregates performance metrics (AUC and accuracy) for the best-trained models. Specify the experiment name, model type, dataset, session, and timestamp.

Output files are saved in:
`results_path/experiment_name/model_to_train/database_name_database_session/timestamp/different_analysis/compute_metrics_from_outputs`

To run the script for EEGNet:
```bash
python analysis/compute_metrics_from_outputs.py --experiment_name "DL_BCI_fairness" --model_to_train "EEGNetv4_SM" --dataset_name "Cho2017" --session 1 --timestamp "CORRESPONDED_TIMESTAMP"
python analysis/compute_metrics_from_outputs.py --experiment_name "DL_BCI_fairness" --model_to_train "EEGNetv4_SM" --dataset_name "Lee2019_MI" --session 1 --timestamp "CORRESPONDED_TIMESTAMP"
python analysis/compute_metrics_from_outputs.py --experiment_name "DL_BCI_fairness" --model_to_train "EEGNetv4_SM" --dataset_name "Lee2019_MI" --session 2 --timestamp "CORRESPONDED_TIMESTAMP"
```
For CSP+LDA:
```bash
python analysis/compute_metrics_from_outputs.py --experiment_name "DL_BCI_fairness" --model_to_train "CSP+LDA" --dataset_name "Cho2017" --session 1 --timestamp "CORRESPONDED_TIMESTAMP"
python analysis/compute_metrics_from_outputs.py --experiment_name "DL_BCI_fairness" --model_to_train "CSP+LDA" --dataset_name "Lee2019_MI" --session 1 --timestamp "CORRESPONDED_TIMESTAMP"
python analysis/compute_metrics_from_outputs.py --experiment_name "DL_BCI_fairness" --model_to_train "CSP+LDA" --dataset_name "Lee2019_MI" --session 2 --timestamp "CORRESPONDED_TIMESTAMP"
```
Executed with the `analysis_environment`.

### 10. `class_distinctiveness/distinctiveness_coefficent_per_subject.py`
This script calculates the class distinctiveness for each subject using the entire session data, following the approach proposed in the manuscript.

To run the script:
```bash
python class_distinctiveness/distinctiveness_coefficent_per_subject.py --script_config "distinctiveness_DL_BCI_fairness_Cho2017"
python class_distinctiveness/distinctiveness_coefficent_per_subject.py --script_config "distinctiveness_DL_BCI_fairness_Lee2019_MI_1"
python class_distinctiveness/distinctiveness_coefficent_per_subject.py --script_config "distinctiveness_DL_BCI_fairness_Lee2019_MI_2"
```
Executed with the `train_environment`.

### 11. `results_DL_BCI_fairness/metrics_tables.py`
This script generates the results tables (accuracy and AUC) published in the manuscript, comparing the CSP+LDA and EEGNet models. Update the timestamps in the script before running it.

To run the script:
```bash
python results_DL_BCI_fairness/metrics_tables.py
```
Executed with the `analysis_environment`.

### 12. `results_DL_BCI_fairness/class_distinctiveness_table.py`
This script generates tables for the class distinctiveness results presented in the manuscript, disaggregated by sex for each dataset and session.

To run the script:
```bash
python results_DL_BCI_fairness/class_distinctiveness_table.py
```
Executed with the `analysis_environment`.

### 13. `results_DL_BCI_fairness/scatterplots_metrics_class_distinctivness.py`
This script generates scatterplots of the performance metrics (accuracy and AUC) versus the class distinctiveness. It produces plots for both the distinctiveness values and their logarithms. One of the resulting plots corresponds to Figure 2 of the manuscript.

To run the script:
```bash
python results_DL_BCI_fairness/scatterplots_metrics_class_distinctivness.py
```
Executed with the `analysis_environment`.

### 14. `results_DL_BCI_fairness/correlation_tables_performance_metrics_class_distinctiveness.py`
This script computes the correlations between performance metrics and class distinctiveness, as reported in the manuscript.

To run the script:
```bash
python results_DL_BCI_fairness/correlation_tables_performance_metrics_class_distinctiveness.py
```
Executed with the `analysis_environment`.

### 13.  results_DL_BCI_fairness/histogram_class_distinctiveness.py

This script computes the histogram of class distinctiveness to understand the decision to take the logarithm of the class distinctiveness.

```bash
python results_DL_BCI_fairness/histogram_class_distinctiveness.py
```

Executed with `analysis_environment`.

### 14. results_DL_BCI_fairness/partial_correlation_tables_metrics_class_distinctiveness.py

This script computes the partial correlation between metrics and class distinctiveness, without counting the effects of sex and age. These results are presented in the original manuscript.

```bash
python results_DL_BCI_fairness/partial_correlation_tables_metrics_class_distinctiveness.py
```

Executed with `analysis_environment`.

### 15. results_DL_BCI_fairness/mixing_model_effects_table.py

This script computes the mixing model effects that allow us to understand the influence of variables such as sex, age, and class distinctiveness without the need to aggregate the performance of each subject, because the model includes the performance of each of the 100 models as a random effect of the subject.

```bash
python results_DL_BCI_fairness/mixing_model_effects_table.py
```

Executed with `analysis_environment`.

### 16. Other scripts that could be interesting to run

The following scripts are not results from the paper but could be useful to understand how the DL model and CSP+LDA model were trained and their performance metrics.

It is important to remember that you must pass the appropriate flags to each script to run them for the respective dataset, session, and timestamp.

- **compute_duration_and_best_epochs.py**: This script works with the DL model and outputs the best epoch of each trained model and the duration of the training.
- **ROC_curves_by_test_subject.py**: This script computes the ROC curves for each trained model. The results are saved in the following folder:

  ```
  results_path/experiment_name/model_to_train/database_name_database_session/timestamp/plots/ROC_curves_by_test_subject
  ```

- **stripplot_and_boxplot_by_test_folder.py**: Creates boxplots and strip plots for each test subject in the training, validation, and test sets. The results are saved in:

  ```
  results_path/experiment_name/model_to_train/database_name_database_session/timestamp/plots/stripplot_and_boxplot_by_test_folder
  ```

- **trainning_curves.py**: This script computes the training and validation curves for each model. The results are saved in:

  ```
  results_path/experiment_name/model_to_train/database_name_database_session/timestamp/plots/trainning_curves
  ```

## Citation


## License
This repository is licensed under [your chosen license, e.g., CC BY-NC-SA 4.0 or MIT], which allows usage with the condition of citation and prohibits modifications.

For more information, please refer to the paper or contact us:[link](bzorzet@sinc.unl.edu.ar) .
