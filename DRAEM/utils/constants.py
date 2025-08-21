CHECKPOINT_DIR = "_fastflow_experiment_checkpoints"

MVTEC_CATEGORIES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

MVTEC_CATEGORIES_PART = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
]

VISA_CATEGORIES = ['candle',
                 'capsules',
                 'cashew',
                 'chewinggum',
                 'fryum',
                 'macaroni1',
                 'macaroni2',
                 'pcb1',
                 'pcb2',
                 'pcb3',
                 'pcb4',
                 'pipe_fryum'
                 ]

BTAD_CATEGORIES = ['btad01',
                 'btad02',
                 'btad03',
                 ]


BACKBONE_DEIT = "deit_base_distilled_patch16_384"
BACKBONE_CAIT = "cait_m48_448"
BACKBONE_RESNET18 = "resnet18"
BACKBONE_WIDE_RESNET50 = "wide_resnet50_2"
EFFICIENTNET_B4 = "efficientnet_b4"

SUPPORTED_BACKBONES = [
    BACKBONE_DEIT,
    BACKBONE_CAIT,
    BACKBONE_RESNET18,
    BACKBONE_WIDE_RESNET50,
]

BATCH_SIZE = 8
NUM_EPOCHS = 1500
LR = 1e-4
WEIGHT_DECAY = 1e-5

LOG_INTERVAL = 10
EVAL_INTERVAL = 10
CHECKPOINT_INTERVAL = 10
OUTPUT_PATH = '/home/scu-its-gpu-001/PycharmProjects/Saliency_AD/checkpoint/'