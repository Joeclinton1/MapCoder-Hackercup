from datasets.Dataset import Dataset
from datasets.MBPPDataset import MBPPDataset
from datasets.APPSDataset import APPSDataset
from datasets.XCodeDataset import XCodeDataset
from datasets.HumanEvalDataset import HumanDataset
from datasets.CodeContestDataset import CodeContestDataset
from datasets.HackercupDataset import HackercupDataset, HackercupDatasetSample # Import the new dataset class


class DatasetFactory:
    @staticmethod
    def get_dataset_class(dataset_name):
        if dataset_name == "APPS":
            return APPSDataset
        elif dataset_name == "MBPP":
            return MBPPDataset
        elif dataset_name == "XCode":
            return XCodeDataset
        elif dataset_name in ["HumanEval", "Human"]:
            return HumanDataset
        elif dataset_name == "CC":
            return CodeContestDataset
        elif dataset_name == "Hackercup":
            return HackercupDataset
        elif dataset_name == "HackercupSample":
            return HackercupDatasetSample
        else:
            raise Exception(f"Unknown dataset name {dataset_name}")
