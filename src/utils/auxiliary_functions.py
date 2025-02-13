def convert_labels_to_int(labels, dict_labels={}):
    import numpy as np
    return np.vectorize(dict_labels.get)(labels)

def build_skorch_kwargs(model_params=None, optimizer_params=None, criterion_params=None):
    kwargs = {}
    # Add model parameters
    if model_params:
        kwargs.update({f'module__{key}': value for key, value in model_params.items()})
    # Add optimizer parameters
    if optimizer_params:
        kwargs.update({f'optimizer__{key}': value for key, value in optimizer_params.items()})
    # Add criterion parameters
    if criterion_params:
        kwargs.update({f'criterion__{key}': value for key, value in criterion_params.items()})
    return kwargs


def zip_files_in_directory(source_directory, output_directory, prefix="compressed_files"):
    import os
    import zipfile
    from datetime import datetime
    """
    Compress all files in the specified source directory into a ZIP file,
    and save it to the specified output directory.
    
    Args:
        source_directory (str): The path to the directory containing the files to compress.
        output_directory (str): The path to the directory where the ZIP file will be saved.
    
    Returns:
        str: The path to the created ZIP file.
    """
    # Create a timestamp for the ZIP file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"{prefix}_{timestamp}.zip"
    zip_filepath = os.path.join(output_directory, zip_filename)

    # Create the ZIP file and add files from the source directory
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, _, files in os.walk(source_directory):
            for file in files:
                full_path = os.path.join(root, file)
                # Add the file to the ZIP file using its relative path to preserve directory structure
                zip_file.write(full_path, os.path.relpath(full_path, source_directory))

    print(f"Files compressed into {zip_filepath}")
    return zip_filepath
