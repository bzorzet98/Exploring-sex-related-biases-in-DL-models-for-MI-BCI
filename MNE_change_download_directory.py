"""
===========================
Change Download Directory
===========================

This is a minimal example to demonstrate how to change the default data download directory to a custom
path/location.
"""
# Authors: Divyesh Narayanan <divyesh.narayanan@gmail.com>
#
# License: BSD (3-clause)
from mne.utils import get_config, set_config
import init_paths
from global_config import DATABASES_PATH, DIRECTORY_TO_SAVE_ROOT

new_path = DATABASES_PATH
original_path = get_config("MNE_DATA")

print(f"The download of MNE_DATA directory is currently {original_path}")
set_config("MNE_DATA", new_path)
path = get_config("MNE_DATA")
print(f"Now the download of MNE_DATA directory is currently {path}")

'MNE_DATASETS_GIGADB_PATH'
original_path = get_config("MNE_DATASETS_GIGADBPATH")
print(f"The download of MNE_DATASETS_GIGADBPATH directory is currently {original_path}")
set_config("MNE_DATASETS_GIGADBPATH", new_path)
path = get_config("MNE_DATASETS_GIGADBPATH")
print(f"Now the download of MNE_DATASETS_GIGADBPATH directory is currently {path}")

original_path = get_config("MNE_DATASETS_GIGADB_PATH")
print(f"The download of MNE_DATASETS_GIGADB_PATH directory is currently {original_path}")
set_config("MNE_DATASETS_GIGADB_PATH", new_path)
path = get_config("MNE_DATASETS_GIGADB_PATH")
print(f"Now the download of MNE_DATASETS_GIGADB_PATH directory is currently {path}")

original_path = get_config("MNE_DATASETS_LEE2019_MI_PATH")
print(f"The download of MNE_DATASETS_LEE2019_MI_PATH directory is currently {original_path}")
set_config("MNE_DATASETS_LEE2019_MI_PATH", new_path)
path = get_config("MNE_DATASETS_LEE2019_MI_PATH")
print(f"Now the download of MNE_DATASETS_LEE2019_MI_PATH directory is currently {path}")

original_path = get_config("MOABB_RESULTS")
print(f"The download of MOABB_RESULTS directory is currently {original_path}")
set_config("MOABB_RESULTS", DIRECTORY_TO_SAVE_ROOT)
path = get_config("MOABB_RESULTS")
print(f"Now the download of MOABB_RESULTS directory is currently {path}")
'MNE_DATASETS_GIGADB_PATH'