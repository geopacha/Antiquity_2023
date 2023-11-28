import os
from os.path import join
import albumentations as A

from rastervision.core.rv_pipeline import *
from rastervision.core.backend import *
from rastervision.core.data import *
from rastervision.core.analyzer import *
from rastervision.pytorch_backend import *
from rastervision.pytorch_learner import *
from rastervision.pytorch_backend.examples.utils import get_scene_info, save_image_crop
from rastervision.PreChippedClassification.prechip_chipclassification import PrechipChipClassification
from rastervision.PreChippedClassification.prechip_chipclassification_config import PrechipChipClassificationConfig
import pdb

#AOI are areas that are presumed to be entirely labeled
#aoi_path = "AOIs/StructuresAOI2.geojson"
def get_config(runner,
        raw_uri:str, 
        processed_uri:str, 
        root_uri:str,
        external_module_uri:str,
        external_model: bool = True, 
        external_loss: bool = False,
        nochip: bool = False,
        test:bool = False) -> PrechipChipClassificationConfig:
    
    debug = False
    
    model_uri = external_module_uri
    train_scene_info = get_scene_info(join(processed_uri,"Scenes",'train-scenes.csv'))
    val_scene_info = get_scene_info(join(processed_uri,"Scenes", 'val-scenes.csv'))
    log_tensorboard = True
#    run_tensorboard = True
    class_config = ClassConfig(names=['no_struct','struct'])


    if test:
        debug = True
        train_scene_info = train_scene_info[0:1]
        val_scene_info = val_scene_info[0:1]

    def make_scene(scene_info) -> SceneConfig:
        (raster_uri, label_uri) = scene_info
        raster_uri = join(raw_uri,"ESPG4326", raster_uri)
        label_uri = join(processed_uri, label_uri)
        # aoi_uri = label_uri
        #aoi_uri = join(raw_uri, aoi_path)

        if test:
            crop_uri = join(processed_uri,'crops',os.path.basename(raster_uri))
            label_crop_uri = join(processed_uri,'crops', os.path.basename(label_uri))
            save_image_crop(
                raster_uri,
                crop_uri,
                label_uri=label_uri,
                label_crop_uri=label_crop_uri,
                size=600,
                min_features=20,
                class_config=class_config)
            raster_uri = crop_uri
            label_uri = label_crop_uri

        id = os.path.splitext((os.path.basename(raster_uri)))[0]
        raster_source = RasterioSourceConfig(
            channel_order=[0,1,2], uris=[raster_uri])
        label_source = ChipClassificationLabelSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(
                uri=label_uri,default_class_id=0, ignore_crs_field=True),
            #ioa_thresh=0.5,
            #use_intersection_over_cell=False,
            pick_min_class_id=False,
            #background_class_id=0,
            infer_cells=False
            #cell_sz=256
        )


        return SceneConfig(
            id=id,
            raster_source=raster_source,
            label_source=label_source,
            #aoi_uris=[aoi_uri]
        )
    # pdb.set_trace()
    chip_sz = 256
    img_sz = chip_sz
    train_scenes = [make_scene(info)for info in train_scene_info]
    val_scenes = [make_scene(info) for info in val_scene_info]
    dataset = DatasetConfig(
        class_config=class_config,
        train_scenes=train_scenes,
        validation_scenes=val_scenes
    )
    if external_model:
        model = ClassificationModelConfig(
                external_def=ExternalModuleConfig(
                    uri=join(external_module_uri,model_uri),
                    name='BYOL_sup_con',
                    entrypoint='byol_sup_con',
                    force_reload=True
                    ))
    else:
        model = ClassificationModelConfig(backbone=Backbone.resnet50)
    data = ClassificationImageDataConfig(img_sz=128, num_workers=8)
    external_loss_def = None

    solver = SolverConfig(
            lr=1e-10,
            num_epochs=2,
            test_num_epochs=3,
            batch_sz=20, 
            one_cycle=True,
            external_loss_def=external_loss_def)
    backend = PyTorchChipClassificationConfig(
        model=model,
        solver=solver,
        data=data,
        log_tensorboard=log_tensorboard,
#        run_tensorboard=run_tensorboard,
        test_mode=test
    )
    # pdb.set_trace()
    config = PrechipChipClassificationConfig(
        root_uri=root_uri,
        dataset=dataset,
        backend=backend,
        train_chip_sz=chip_sz,
        predict_chip_sz=chip_sz
    )
    return config
