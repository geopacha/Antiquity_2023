from rastervision.pipeline.config import register_config, ConfigError
from rastervision.core.rv_pipeline.chip_classification_config import ChipClassificationConfig
from rastervision.core.data.label_store import (
    ChipClassificationGeoJSONStoreConfig)
from rastervision.core.evaluation import ChipClassificationEvaluatorConfig

@register_config('prechip_chipclassification')
class PrechipChipClassificationConfig(ChipClassificationConfig):
    def build(self, tmp_dir):
        from rastervision.PreChippedClassification.prechip_chipclassification import PrechipChipClassification
        return PrechipChipClassification(self, tmp_dir)

    def validate_config(self):
        if self.train_chip_sz != self.predict_chip_sz:
            raise ConfigError(
                'train_chip_sz must be equal to predict_chip_sz for chip '
                'classification.')

    def get_default_label_store(self, scene):
        return ChipClassificationGeoJSONStoreConfig()

    def get_default_evaluator(self):
        return ChipClassificationEvaluatorConfig()
