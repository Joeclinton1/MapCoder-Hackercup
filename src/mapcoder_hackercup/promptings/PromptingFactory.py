from mapcoder_hackercup.promptings.CoT import CoTStrategy
from mapcoder_hackercup.promptings.Direct import DirectStrategy
from mapcoder_hackercup.promptings.Analogical import AnalogicalStrategy
from mapcoder_hackercup.promptings.SelfPlanning import SelfPlanningStrategy

from mapcoder_hackercup.promptings.MapCoder import MapCoder as MapCoder
from mapcoder_hackercup.promptings.Custom import Custom

class PromptingFactory:
    @staticmethod
    def get_prompting_class(prompting_name):
        if prompting_name == "CoT":
            return CoTStrategy
        elif prompting_name == "MapCoder":
            return MapCoder
        elif prompting_name == "Direct":
            return DirectStrategy
        elif prompting_name == "Analogical":
            return AnalogicalStrategy
        elif prompting_name == "SelfPlanning":
            return SelfPlanningStrategy
        elif prompting_name == "Custom":
            return Custom
        else:
            raise Exception(f"Unknown prompting name {prompting_name}")
