import argparse
import time 
import cv2
import numpy as np
from models.common import OpenvinoInference, OnnxInference

 
class SegPipeline:
    def __init__(self, segmenter, onnx_model, classes, imgsz=(512, 512), infer_tool='openvino'):
        self.segmenter = segmenter
        self.model_height, self.model_width = imgsz[0], imgsz[1]  # 图像resize大小
        self.classes =  classes

        self.infer_tool = infer_tool
        if self.infer_tool == 'openvino':
            # 构建openvino推理引擎
            self.openvino = OpenvinoInference(onnx_model)
            self.ndtype = self.openvino.ndtype
        else:
            # 构建onnxruntime推理引擎
            self.ort_session = OnnxInference(onnx_model)
            self.ndtype = self.ort_session.ndtype
       
        

    def __call__(self, im0, vis_masks=False):
        """
        The whole pipeline: pre-process -> inference -> post-process.
        Args:
            im0 (Numpy.ndarray): original input image.
        Returns:
            masks (Numpy.ndarray).
        """
        # 前处理 Pre-process  
        t1 = time.time()
        im, im0_shape = self.preprocess(im0)
        print('seg预处理时间: {:.3f}s'.format(time.time() - t1))
        
        # 推理 Inference
        t2 = time.time()
        if self.infer_tool == 'openvino':
            preds = self.openvino.predict(im) 
        else:
            preds = self.ort_session.predict(im) 
        
        # 后处理 Postprocess
        masks = self.segmenter.postprocess(preds)  # (1024, 1024) 
        masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))[:, :, None]  # (1024, 1024) --> (3000, 4000, 1)
        print('seg推理+后处理时间：{:.3f}s'.format(time.time() - t2))  
    
        if vis_masks:
            self.vis_masks(np.squeeze(masks), im0_shape)

        return masks


    # 图片预处理
    def preprocess(self, image):
        shape = image.shape[:2]  # original image shape
        img = cv2.resize(image, (self.model_width, self.model_height), interpolation=cv2.INTER_LINEAR)

        # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
        img = np.ascontiguousarray(np.einsum('HWC->CHW', img)[::-1], dtype=self.ndtype) / 255.0
        #CHW---> 1,C,H,W
        img = img[None] if len(img.shape) == 3 else img
        return img, shape

    
    # 可视化各
    def vis_masks(self, predictions, im0_shape):
        num_classes = len(self.classes.METAINFO['classes'])
        ids = np.unique(predictions)[::-1]  # 找出sem_seg数组中的所有唯一元素（即所有唯一的类别标签） [3 2 1 0]
        legal_indices = ids < num_classes  # 确保只处理有效的类别标签，即那些在classes列表范围内的标签
        ids = ids[legal_indices]  # 使用legal_indices来过滤ids，只保留那些有效的类别标签
        labels = np.array(ids, dtype=np.int64)
        colors = [self.classes.METAINFO['palette'][label] for label in labels]  # 从palette字典或列表中获取与每个label对应的颜色
        mask = np.zeros_like(np.zeros((im0_shape[0], im0_shape[1], 3)), dtype=np.uint8)

        # 使用条件索引来查找preds[0][0]中所有等于当前label的位置，并将这些位置的mask数组的值设置为对应的color
        for label, color in zip(labels, colors):
            mask[predictions == label, :] = color  # 只对第一行（或第一个平面）进行处理就行
        cv2.imwrite('./outputs/vis_masks.jpg', mask)
