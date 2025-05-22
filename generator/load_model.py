from detectron2.modeling import build_model
import sys
sys.path.insert(1, '../third_party/CenterNet2/projects/CenterNet2/')
sys.path.insert(2, '..')
sys.path.insert(3, '../odod/modeling/meta_arch/')
from centernet.config import add_centernet_config
from mrca.config import add_bsgal_config
from detectron2.config import get_cfg

from detectron2.engine import (default_argument_parser, default_setup)
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

from detectron2.checkpoint import DetectionCheckpointer
import torch


def setup(cfg_path):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_bsgal_config(cfg)
    cfg.merge_from_file(cfg_path)
    # cfg.merge_from_list(args.opts)
    if '/auto' in cfg.OUTPUT_DIR:
        file_name = os.path.basename(cfg_path)[:-5]

        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))

        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    if '/amlt' in cfg.OUTPUT_DIR:
        file_name = os.environ.get('AMLT_OUTPUT_DIR','OUTPUT')
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/amlt', file_name)
        logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))
    cfg.freeze()

    return cfg



def load_model(cfg_path, ckptPath):
    cfg = setup(cfg_path)
    meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # edit meta
    if cfg.INPUT.SEPARATE_SYN and not cfg.INPUT.SEPERATE_SUP:
        for i in range(len(meta.thing_classes)):
            meta.class_image_count.append({'id':i+1204,'image_count':2000})

    model = build_model(cfg)

    if cfg.SOLVER.MODEL_EMA > 0:
        import tempfile
        tmp = torch.load(ckptPath, map_location='cpu')
        tmp['model'] = tmp['model_ema']
        tmp_file = tempfile.NamedTemporaryFile()
        torch.save(tmp, tmp_file.name)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            tmp_file.name, resume=False
        )
    else :
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            ckptPath, resume=False
        )

    return model


