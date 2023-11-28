import logging

import numpy as np
import pdb
from typing import List
from rastervision.core.rv_pipeline.chip_classification import ChipClassification
from rastervision.core.box import Box

log = logging.getLogger(__name__)



def get_train_windows(scene):
    # pdb.set_trace()
    labels = scene.ground_truth_label_source.get_labels()
    train_windows=labels.get_cells()
    return train_windows

def get_predict_windows(self, extent: Box) -> List[Box]:
    """Returns windows to compute predictions for.
    Args:
        extent: extent of RasterSource
    """
    chip_sz = stride = self.config.predict_chip_sz
    return extent.get_windows(chip_sz, stride)

class PrechipChipClassification(ChipClassification):
    commands = ['predict','eval','bundle']
    def get_train_windows(self,scene):
        return get_train_windows(scene)
