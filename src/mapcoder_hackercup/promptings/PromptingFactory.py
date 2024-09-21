from mapcoder_hackercup.promptings.CoT import CoTStrategy
from mapcoder_hackercup.promptings.Direct import DirectStrategy

from mapcoder_hackercup.promptings.MapCoder import MapCoder as MapCoder
from mapcoder_hackercup.promptings.Custom import Custom
from mapcoder_hackercup.promptings.MapCoder_CodeGenOnly import CodeGenOnly
from mapcoder_hackercup.promptings.Custom_DirectPlanning import DirectPlanning

class PromptingFactory:
    @staticmethod
    def get_prompting_class(prompting_name):
        if prompting_name == "CoT":
            return CoTStrategy
        elif prompting_name == "MapCoder":
            return MapCoder
        elif prompting_name == "Direct":
            return DirectStrategy
        elif prompting_name == "Custom":
            return Custom
        elif prompting_name == "CodeGenOnly":
            return CodeGenOnly
        elif prompting_name == "DirectPlanning":
            return DirectPlanning
        else:
            raise Exception(f"Unknown prompting name {prompting_name}")
