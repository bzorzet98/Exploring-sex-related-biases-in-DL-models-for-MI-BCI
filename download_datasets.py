import argparse
import init_paths
from src.utils.imports import import_class
from moabb.datasets import Shin2017A, Shin2017B
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_to_download', type=str, default = 'Shin2017A')
parser.add_argument('--from_moabb', type=bool, default = True)
args = parser.parse_args()

if args.from_moabb:
    module_name = 'moabb.datasets'
    dataset_class = import_class(args.dataset_to_download,module_name=module_name)
    if args.dataset_to_download in ['Shin2017A', 'Shin2017B']:
        dataset = dataset_class(accept=True)
    else:
        dataset = dataset_class()
    subject_list = dataset.subject_list
    print(f"Subject list: {subject_list}")
    dataset.get_data(subject_list) 

