from mapcoder_hackercup.promptings.CoT import CoTStrategy
from mapcoder_hackercup.promptings.Direct import DirectStrategy

from mapcoder_hackercup.promptings.MapCoder import MapCoder as MapCoder
from mapcoder_hackercup.promptings.Custom import Custom
from mapcoder_hackercup.promptings.Custom_DirectPlanning import DirectPlanning
from mapcoder_hackercup.promptings.Matus import Matus
from mapcoder_hackercup.promptings.Matus_ParallelCode import ParallelCode
from mapcoder_hackercup.promptings.Joe import Joe
from mapcoder_hackercup.promptings.Zac import Zac
from mapcoder_hackercup.promptings.Baseline import Baseline

class PromptingFactory:
    @staticmethod
    def get_prompting_class(prompting_name):
        match prompting_name:
            case "CoT":
                return CoTStrategy
            case "MapCoder":
                return MapCoder
            case "Direct":
                return DirectStrategy
            case "Custom":
                return Custom
            case "DirectPlanning":
                return DirectPlanning
            case "Matus":
                return Matus
            case "ParallelCode":
                return ParallelCode
            case "Joe":
                return Joe
            case "Zac":
                return Zac
            case "Baseline":
                return Baseline
            case _:
                raise Exception(f"Unknown prompting name {prompting_name}")
