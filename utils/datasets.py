def Get_datasets(dataset_name):
    # 使用 globals() 获取类  
    cls = globals().get(dataset_name)  
    
    # 确保我们找到了一个类（而不是 None 或其他非类对象）  
    if cls is not None and issubclass(cls, object):  
        # 创建类的实例  
        instance = cls()  
        return instance
    else:  
        print(f"数据集： {dataset_name} not found， 名字不对或不存在！")
        return False


class Custom():
    def __init__(self):
        self.METAINFO = dict(
            classes=('background', 'jdc', 'fjdc', 'rxd'),
            palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0]])



class PascalVOC():
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    def __init__(self):
        self.METAINFO = dict(
            classes=('background', 'aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                    'sofa', 'train', 'tvmonitor'),
            palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                    [0, 64, 128]])


class Cityscapes():
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    def __init__(self):
        self.METAINFO = dict(
        classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                 'traffic light', 'traffic sign', 'vegetation', 'terrain',
                 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                 'motorcycle', 'bicycle'),
        palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])
    

class COCOStuff():
    """COCO-Stuff dataset.

    In segmentation map annotation for COCO-Stuff, Train-IDs of the 10k version
    are from 1 to 171, where 0 is the ignore index, and Train-ID of COCO Stuff
    164k is from 0 to 170, where 255 is the ignore index. So, they are all 171
    semantic categories. ``reduce_zero_label`` is set to True and False for the
    10k and 164k versions, respectively. The ``img_suffix`` is fixed to '.jpg',
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    def __init__(self):
        self.METAINFO = dict(
        classes=(
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
            'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
            'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
            'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
            'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
            'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower',
            'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel',
            'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
            'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
            'paper', 'pavement', 'pillow', 'plant-other', 'plastic',
            'platform', 'playingfield', 'railing', 'railroad', 'river', 'road',
            'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf',
            'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs',
            'stone', 'straw', 'structural-other', 'table', 'tent',
            'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick',
            'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone',
            'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
            'window-blind', 'window-other', 'wood'),
        palette=[[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
                 [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
                 [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
                 [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
                 [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
                 [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
                 [64, 128, 32], [0, 160, 0], [0, 0, 0], [192, 128, 160],
                 [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0],
                 [0, 128, 0], [192, 128, 32], [128, 96, 128], [0, 0, 128],
                 [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160],
                 [0, 96, 128], [128, 128, 128], [64, 0, 160], [128, 224, 128],
                 [128, 128, 64], [192, 0, 32], [128, 96, 0], [128, 0, 192],
                 [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160],
                 [64, 96, 0], [0, 128, 192], [0, 128, 160], [192, 224, 0],
                 [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
                 [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160],
                 [64, 32, 128], [128, 192, 192], [0, 0, 160], [192, 160, 128],
                 [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128],
                 [64, 128, 96], [64, 160, 0], [0, 64, 0], [192, 128, 224],
                 [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
                 [0, 192, 0], [192, 128, 96], [192, 96, 128], [0, 64, 128],
                 [64, 0, 96], [64, 224, 128], [128, 64, 0], [192, 0, 224],
                 [64, 96, 128], [128, 192, 128], [64, 0, 224], [192, 224, 128],
                 [128, 192, 64], [192, 0, 96], [192, 96, 0], [128, 64, 192],
                 [0, 128, 96], [0, 224, 0], [64, 64, 64], [128, 128, 224],
                 [0, 96, 0], [64, 192, 192], [0, 128, 224], [128, 224, 0],
                 [64, 192, 64], [128, 128, 96], [128, 32, 128], [64, 0, 192],
                 [0, 64, 96], [0, 160, 128], [192, 0, 64], [128, 64, 224],
                 [0, 32, 128], [192, 128, 192], [0, 64, 224], [128, 160, 128],
                 [192, 128, 0], [128, 64, 32], [128, 32, 64], [192, 0, 128],
                 [64, 192, 32], [0, 160, 64], [64, 0, 0], [192, 192, 160],
                 [0, 32, 64], [64, 128, 128], [64, 192, 160], [128, 160, 64],
                 [64, 128, 0], [192, 192, 32], [128, 96, 192], [64, 0, 128],
                 [64, 64, 32], [0, 224, 192], [192, 0, 0], [192, 64, 160],
                 [0, 96, 192], [192, 128, 128], [64, 64, 160], [128, 224, 192],
                 [192, 128, 64], [192, 64, 32], [128, 96, 64], [192, 0, 192],
                 [0, 192, 32], [64, 224, 64], [64, 0, 64], [128, 192, 160],
                 [64, 96, 64], [64, 128, 192], [0, 192, 160], [192, 224, 64],
                 [64, 128, 64], [128, 192, 32], [192, 32, 192], [64, 64, 192],
                 [0, 64, 32], [64, 160, 192], [192, 64, 64], [128, 64, 160],
                 [64, 32, 192], [192, 192, 192], [0, 64, 160], [192, 160, 192],
                 [192, 192, 0], [128, 64, 96], [192, 32, 64], [192, 64, 128],
                 [64, 192, 96], [64, 160, 64], [64, 64, 0]])