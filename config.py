import albumentations as A
from albumentations.pytorch import ToTensorV2
from random_word import RandomWords
import warnings
import cv2
warnings.filterwarnings("ignore")
r = RandomWords()


class CFG:
    # exp
    comp_name = 'vesuvius'
    comp_dir_path = './'
    comp_folder_name = 'data'
    comp_dataset_path = f'{comp_dir_path}{comp_folder_name}/'
    exp_name = "pretrained" # set your desired experiment name

    # Models
    model = 'ink-classifier' # 2.5D, 3D-2D, ink-classifier, 3D-3D-1D
    slices = (20, 36)
    in_chans = slices[1] - slices[0]
    target_size = 1
    pretrained_dir = "./weights/pretrained/vesuvius-models/"
    use_pretrained = False
    use_denoiser = True

    # 2.5D
    encoder = 'seresnext50_32x4d'
    decoder = 'Unet' # Unet, Unet++, DeepLabV3+
    depth = 4
    decoder_channels = [256, 128, 64, 32]
    drop_rate = 0.3
    drop_path_rate = 0.2
    use_pool = True

    # 3D-2D
    pooler = "conv_attention" # projection, attention, mean, conv, conv_attention, max
    encoder3d = "resnet" # 2D, resnet, custom
    decoder3d2d = "FPN" # FPN, Custom
    encoder_depth = 34
    weight_path3d = f"r3d{encoder_depth}_KM_200ep.pth"

    # ink-classifier
    region_size = 64
    region_stride = region_size//2
    feature_pooler = "mean" # mean, max, None, gem
    
    # 3D-3D-1D
    pooler3d = "attention" # attention, mean, max, unet, cls, conv_attention
    
    # training
    size = 224
    tile_size = 224
    freeze = False
    use_pretrained_model = False
    sampling_method = "adaptive_stride_random"
    filter_method = "mean"
    stride = tile_size // 4
    valid_stride = tile_size // 4
    neg_sample_size = 8
    n_folds = 5
    folds = range(1, n_folds + 1)
    train_dir = f"/media/fql/Data/Kaggle/Vesuvius Challenge - Ink Detection/data/sampled/{size}*{size}_stride{stride}_negsample{neg_sample_size}__slice{slices}method({sampling_method})/"
    train_batch_size = 4
    valid_batch_size = train_batch_size
    use_amp = True
    grad_clipping = True
    llrd = False
    optimizer = "AdamW" # AdamW, Lion

    scheduler = "cosine"
    num_cycles = 0.5
    num_warmup_steps_rate = 0.1
    epochs = 20

    if scheduler == "gradualwarmup":
        warmup_factor = 10
        lr = 1e-4 / warmup_factor
    else:
        lr = 1e-4
    loss = "DiceBCE"

    pretrained = True
    inf_weight = 'best'  # 'best'
    min_lr = 1e-6
    weight_decay = 1e-3
    max_grad_norm = 1000
    num_workers = 4
    seed = 42

    # output
    outputs_path = f'./weights/{exp_name}/'
    model_dir = outputs_path + f'{comp_name}-models/'
    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}'

    # augmentations
    train_aug_list = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(rotate=0, translate_percent=0.1, scale=[0.9,1.5], shear=0, p=0.5),
        A.OneOf([
            A.RandomToneCurve(scale=0.3, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.4, 0.5), brightness_by_max=True, always_apply=False, p=0.8)
        ], p=0.5),
        A.OneOf([
            A.ShiftScaleRotate(shift_limit=None, scale_limit=[-0.15, 0.15], rotate_limit=[-30, 30], interpolation=cv2.INTER_LINEAR,border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=None, shift_limit_x=[-0.1, 0.1],shift_limit_y=[-0.2, 0.2], rotate_method='largest_box', p=0.5),
            A.ElasticTransform(alpha=1, sigma=20, alpha_affine=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=None, approximate=False, same_dxdy=False, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=None, normalized=True, p=0.5),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=[10, 50], p=0.5),
            A.GaussianBlur(p=0.5),
            A.MotionBlur(p=0.5),
        ], p=0.5),
        A.CoarseDropout(max_holes=3, max_width=0.15, max_height=0.25, mask_fill_value=0, p=0.5),
        A.Normalize(
            mean=[0]*in_chans, 
            std=[1]*in_chans, 
        ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Normalize(
            mean=[0]*in_chans, 
            std=[1]*in_chans, 
        ),
        ToTensorV2(transpose_mask=True),
    ]

    tta = False
