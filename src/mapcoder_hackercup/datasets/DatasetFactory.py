from mapcoder_hackercup.datasets.Dataset import Dataset
from mapcoder_hackercup.datasets.MBPPDataset import MBPPDataset
from mapcoder_hackercup.datasets.APPSDataset import APPSDataset
from mapcoder_hackercup.datasets.XCodeDataset import XCodeDataset
from mapcoder_hackercup.datasets.HumanEvalDataset import HumanDataset
from mapcoder_hackercup.datasets.CodeContestDataset import CodeContestDataset
from mapcoder_hackercup.datasets.HackercupDataset import HackercupDataset, HackercupDatasetSample # Import the new dataset class


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
