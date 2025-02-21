# Exploring Sex-Related Biases in DL Models for MI-BCI

This repository contains the code and resources necessary to reproduce the experiments presented in the paper *"Exploring Sex-Related Biases in Deep Learning Models for Motor Imagery Brain-Computer Interfaces"*. The provided scripts cover data preprocessing, model training, evaluation, and analysis.

## Dowload this repository
Clone this repository:
```bash
  git clone https://github.com/bzorzet98/Exploring-sex-related-biases-in-DL-models-for-MI-BCI.git

  cd Exploring-sex-related-biases-in-DL-models-for-MI-BCI
```

## Index
Index
- Environment Setup
  1. Train Environment
  2.  EEG Environment
  3.  Analysis Environment
- General Comments
- Database Configuration
  * Changing the Database Path
- Scripts Description
  1. `global_config.py`
  2. `init_paths.py`
  3. `MNE_change_download_directory.py`
  4. `download_datasets.py`
  5. `metadata_simple_analysis.py`
  6. `preprocessing_datasets/preprocessing_datasets.py`
  7. `train_LOSO_balanced_by_sex/train_LOSO_balanced_by_sex.py`
  8. `train_LOSO_balanced_by_sex_CSP+LDA/train_LOSO_balanced_by_sex_CSP+LDA.py`
  9. `analysis/compute_metrics_from_outputs.py`
  10. `class_distinctiveness/distinctiveness_coefficent_per_subject.py`
  11. `results_DL_BCI_fairness/metrics_tables.py`
  12. `results_DL_BCI_fairness/class_distinctiveness_table.py`
  13. `results_DL_BCI_fairness/scatterplots_metrics_class_distinctivness.py`
  14. `results_DL_BCI_fairness/correlation_tables_performance_metrics_class_distinctiveness.py`
  15. `results_DL_BCI_fairness/histogram_class_distinctiveness.py`
  16. `results_DL_BCI_fairness/partial_correlation_tables_metrics_class_distinctiveness.py`
  17. `results_DL_BCI_fairness/mixing_model_effects_table.py`
  18. Other scripts that could be interesting to run
5. Possible bugs
6. Citation 
7. License

## Environment Setup
To ensure reproducibility, the following Conda environments need to be created with their respective libraries.

### 1. Train Environment
This environment is used for model training and evaluation.
```bash
conda create --name train_environment python=3.11.9
```
For install the packages required run this commands:
```bash
conda activate train_environment
conda install -c conda-forge scipy=1.14
conda install -c conda-forge scikit-learn
```
You can install pytorch without CUDA support: 
```bash
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 -c pytorch
```
Or with CUDA support:
```bash
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
Continue installing the following packages:
```bash
conda install -c conda-forge pandas
python -m pip install -U skorch
pip install einops
conda install -c conda-forge pyriemann
```
### 2. EEG Environment
This environment is required for EEG data preprocessing.
```bash
conda create --name eeg_environment python=3.11.9
```
For install the packages required run this commands:

```bash
conda activate eeg_environment
conda install -c conda-forge scipy=1.14
conda install --channel conda-forge --strict-channel-priority mne
pip install pandas
pip install moabb==1.1.1
```

### 3. Analysis Environment
This environment is used for results analysis, plotting, and metric computation.
```bash
conda create --name analysis_environment python=3.11.9
```
For install the packages required run this commands:
```bash
conda activate analysis_environment 
conda install -c conda-forge seaborn
conda install pandas
pip install pingouin
pip install scikit-learn
```
## General Comments
Due to the large volume of data, GitHub does not allow uploading all result files. Additionally, because of the large number of trained models, their parameters are not included in this repository. However, you can access all the analysis outputs derived from the model results via this [link](https://drive.google.com/drive/folders/1xlziR2qLWgLGcgJ8t3Nzqug0Nb_PVPzX?usp=sharing).
## Database Configuration  

By default, datasets are downloaded to the default directory used by the `moabb` package. However, you can specify a custom download directory if needed.  

In the repository, there is a folder named **EEG_DATABASES**, which contains a subfolder **database_information**. This subfolder includes essential metadata that must remain in the same directory where the datasets will be downloaded.  

To set or modify the download directory, you need to update the `DATABASES_PATH` variable in the **global_config.py** file. This path must:  
- Be the location where the EEG databases will be downloaded.  
- Contain the **EEG_DATABASES/database_information/** folder.  
- Store the preprocessed versions of the datasets.  

### Changing the Database Path  
If you want to change the download path, make sure to:  
1. Update the `DATABASES_PATH` variable in **global_config.py**.  
2. Move or add the **database_information/** folder to the new path.  

## Scripts Description  

In this section, we provide a general description of each Python script in this repository and explain how to run them.  

### 1. `global_config.py`  
This Python file contains predefined dictionaries and path configurations to manage results. By default, datasets and results are stored in the same repository folder, but you can configure them as needed. For dataset changes, please refer to the **Database Configuration** section above.  

### 2. `init_paths.py`  
This script configures the system paths by adding the appropriate source files depending on the environment. It is typically called at the beginning of most scripts.  

### 3. `MNE_change_download_directory.py`  
This script is necessary for redefining the paths used by MNE, which is required to change the default path when downloading datasets from MOABB. Running this script is optional but recommended due to the large volume of data.  

To run it, execute:  
```bash
conda activate eeg_environment  
python MNE_change_download_directory.py
```
### 4. `download_datasets.py`
This script downloads the specified dataset using the `--dataset_to_download` flag. To download the datasets 'Cho2017' and 'Lee2019_MI'.
To run it, execute:

```bash
conda activate eeg_environment
python download_datasets --dataset_to_download "Cho2017"
python download_datasets --dataset_to_download "Lee2019_MI"
```

### 5. `metadata_simple_analysis.py`
This script processes the metadata of the dataset, selecting specific column names for analysis.
```bash
conda activate analysis_environment
python metadata_simple_analysis.py --dataset_name 'Cho2017'
python metadata_simple_analysis.py --dataset_name 'Lee2019_MI'
```

### 6. `preprocessing_datasets/preprocessing_datasets.py`
This script preprocesses the data from the databases using configuration files located in the `configs` folder. The preprocessing generates two `.npy` files per subject, session, and run (if applicable). These files are saved by default in `EEG_DATABASES/preprocessed_databases`. 

To run the script:
```bash
conda activate eeg_environment
python preprocessing_datasets/preprocessing_datasets.py --script_config 'classical_preprocessing_Cho2017'
python preprocessing_datasets/preprocessing_datasets.py --script_config 'classical_preprocessing_Lee2019_MI'
```
### 7. `train_LOSO_balanced_by_sex/train_LOSO_balanced_by_sex.py`

This script trains a model defined in the config file using a **Leave-One-Subject-Out (LOSO)** scheme, ensuring that the training and validation sets are balanced by sex. The config file contains various parameters to customize the training process.

The output files will be saved in the following directory:

`results_path/experiment_name/model_to_train/database_name_database_session/timestamp`


If training is interrupted, you can resume it by specifying the generated timestamp in the config file.

#### To run the script with the same configuration as in the paper, execute the following commands:  
(Note: The number of models to train will be 100 per test subject, so this may take considerable time.)
```bash
conda activate train_environment  
python train_LOSO_balanced_by_sex/train_LOSO_balanced_by_sex.py --script_config 'EEGNetv4_Cho2017_paper_config'  
python train_LOSO_balanced_by_sex/train_LOSO_balanced_by_sex.py --script_config 'EEGNetv4_Lee2019_MI_1_paper_config'  
python train_LOSO_balanced_by_sex/train_LOSO_balanced_by_sex.py --script_config 'EEGNetv4_Lee2019_MI_2_paper_config'  
```
#### Alternatively, you can configure the script to train only one model per test subject with the following commands:
```bash
conda activate train_environment  
python train_LOSO_balanced_by_sex/train_LOSO_balanced_by_sex.py --script_config 'EEGNetv4_Cho2017_simple_config'  
python train_LOSO_balanced_by_sex/train_LOSO_balanced_by_sex.py --script_config 'EEGNetv4_Lee2019_MI_1_simple_config'  
python train_LOSO_balanced_by_sex/train_LOSO_balanced_by_sex.py --script_config 'EEGNetv4_Lee2019_MI_2_simple_config'  
```
Output files structure:
For each test subject in the dataset, a separate folder is created for each trained model. The folder is named using the following format:

`split_seed_X_splitseed_model_init_seed_Y`

Inside each folder, the following files will be generated:
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
conda activate eeg_environment
python train_LOSO_balanced_by_sex_CSP+LDA/train_LOSO_balanced_by_sex_CSP+LDA.py --script_config 'CSP+LDA_Cho2017_paper_config'
python train_LOSO_balanced_by_sex_CSP+LDA/train_LOSO_balanced_by_sex_CSP+LDA.py --script_config 'CSP+LDA_Lee2019_MI_1_paper_config'
python train_LOSO_balanced_by_sex_CSP+LDA/train_LOSO_balanced_by_sex_CSP+LDA.py --script_config 'CSP+LDA_Lee2019_MI_2_paper_config'
```

### 9. `analysis/compute_metrics_from_outputs.py`
This script calculates and aggregates performance metrics (AUC and accuracy) for the best-trained models. Specify the experiment name, model type, dataset, session, and timestamp.

Output files are saved in:
`results_path/experiment_name/model_to_train/database_name_database_session/timestamp/different_analysis/compute_metrics_from_outputs`

To run the script for EEGNet:
```bash
conda activate analysis_environment
python analysis/compute_metrics_from_outputs.py --experiment_name "DL_BCI_fairness" --model_to_train "EEGNetv4_SM" --dataset_name "Cho2017" --session 1 --timestamp "CORRESPONDED_TIMESTAMP"
python analysis/compute_metrics_from_outputs.py --experiment_name "DL_BCI_fairness" --model_to_train "EEGNetv4_SM" --dataset_name "Lee2019_MI" --session 1 --timestamp "CORRESPONDED_TIMESTAMP"
python analysis/compute_metrics_from_outputs.py --experiment_name "DL_BCI_fairness" --model_to_train "EEGNetv4_SM" --dataset_name "Lee2019_MI" --session 2 --timestamp "CORRESPONDED_TIMESTAMP"
```
For CSP+LDA:
```bash
conda activate analysis_environment
python analysis/compute_metrics_from_outputs.py --experiment_name "DL_BCI_fairness" --model_to_train "CSP+LDA" --dataset_name "Cho2017" --session 1 --timestamp "CORRESPONDED_TIMESTAMP"
python analysis/compute_metrics_from_outputs.py --experiment_name "DL_BCI_fairness" --model_to_train "CSP+LDA" --dataset_name "Lee2019_MI" --session 1 --timestamp "CORRESPONDED_TIMESTAMP"
python analysis/compute_metrics_from_outputs.py --experiment_name "DL_BCI_fairness" --model_to_train "CSP+LDA" --dataset_name "Lee2019_MI" --session 2 --timestamp "CORRESPONDED_TIMESTAMP"
```

### 10. `class_distinctiveness/distinctiveness_coefficent_per_subject.py`
This script calculates the class distinctiveness for each subject using the entire session data, following the approach proposed in the manuscript.

To run the script:
```bash
conda activate train_environment
python class_distinctiveness/distinctiveness_coefficent_per_subject.py --script_config "distinctiveness_DL_BCI_fairness_Cho2017"
python class_distinctiveness/distinctiveness_coefficent_per_subject.py --script_config "distinctiveness_DL_BCI_fairness_Lee2019_MI_1"
python class_distinctiveness/distinctiveness_coefficent_per_subject.py --script_config "distinctiveness_DL_BCI_fairness_Lee2019_MI_2"
```

### 11. `results_DL_BCI_fairness/metrics_tables.py`
This script generates the results tables (accuracy and AUC) published in the manuscript, comparing the CSP+LDA and EEGNet models. Update the timestamps in the script before running it. But if you downloaded the results from the link of google drive, you can run with the original timestamps

To run the script:
```bash
conda activate analysis_environment
python results_DL_BCI_fairness/metrics_tables.py
```

### 12. `results_DL_BCI_fairness/class_distinctiveness_table.py`
This script generates tables for the class distinctiveness results presented in the manuscript, disaggregated by sex for each dataset and session.

To run the script:
```bash
conda activate analysis_environment
python results_DL_BCI_fairness/class_distinctiveness_table.py
```

### 13. `results_DL_BCI_fairness/scatterplots_metrics_class_distinctivness.py`
This script generates scatterplots of the performance metrics (accuracy and AUC) versus the class distinctiveness. It produces plots for both the distinctiveness values and their logarithms.

To run the script:
```bash
conda activate analysis_environment
python results_DL_BCI_fairness/scatterplots_metrics_class_distinctivness.py
```

### 14. `results_DL_BCI_fairness/correlation_tables_performance_metrics_class_distinctiveness.py`
This script computes the correlations between performance metrics and class distinctiveness, as reported in the manuscript.

To run the script:
```bash
conda activate analysis_environment
python results_DL_BCI_fairness/correlation_tables_performance_metrics_class_distinctiveness.py
```

### 15.  results_DL_BCI_fairness/histogram_class_distinctiveness.py

This script computes the histogram of class distinctiveness to understand the decision to take the logarithm of the class distinctiveness.

```bash
conda activate analysis_environment
python results_DL_BCI_fairness/histogram_class_distinctiveness.py
```

### 16. results_DL_BCI_fairness/partial_correlation_tables_metrics_class_distinctiveness.py

This script computes the partial correlation between metrics and class distinctiveness, without counting the effects of sex and age. These results are presented in the original manuscript.

```bash
conda activate analysis_environment
python results_DL_BCI_fairness/partial_correlation_tables_metrics_class_distinctiveness.py
```

### 17. results_DL_BCI_fairness/mixing_model_effects_table.py

This script computes the mixing model effects that allow us to understand the influence of variables such as sex, age, and class distinctiveness without the need to aggregate the performance of each subject, because the model includes the performance of each of the 100 models as a random effect of the subject.

```bash
conda activate analysis_environment
python results_DL_BCI_fairness/mixing_model_effects_table.py
```

### 18. Other scripts that could be interesting to run

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

## Posibles Bugs

## Citation


## License
This repository is licensed under [your chosen license, e.g., CC BY-NC-SA 4.0 or MIT], which allows usage with the condition of citation and prohibits modifications.

For more information, please refer to the paper or contact us:[link](bzorzet@sinc.unl.edu.ar) .
