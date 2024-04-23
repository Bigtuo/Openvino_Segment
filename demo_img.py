import cv2
import time
import argparse
from utils.datasets import Get_datasets
from models.seg_pipeline import SegPipeline


'''
注意：
1、确定Netron查看onnx文件, 确定模型输入输出维度,是动态/静态输入,图像输入尺寸
2、确定自己训练数据集, 可在对应文件自定义类别与掩码颜色
3、确定前/后处理操作, 前处理是直接归一化还是标准化, 后处理输出的第二维度是多少
'''

if __name__ == '__main__':
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=str('img/demo.png'), help='Path to input image')
    parser.add_argument('--imgsz', type=tuple, default=(512, 1024), help='Image input size, (H, W)')
    parser.add_argument('--datasets', type=str, default='Cityscapes', choices=("Custom", "Cityscapes"), help='模型训练的数据集')
    parser.add_argument('--model_onnx', type=str, default='weights\\stdcnet.onnx', help='Path to ONNX model')

    parser.add_argument('--preprocess_type', type=str, default='Standardization', choices=("Normalization", "Standardization"),
                         help='分割前处理操作, 表示图像前处理采用是直接除255归一化缩放,还是标准化处理, mmseg-0.3版默认为Standardization')
    parser.add_argument('--postprocess_type', type=str, default='One', choices=("One", "Mutil"),
                         help='分割前处理操作, 表示模型输出的第二个维度是一维还是多维, mmseg-0.3版默认为一维')
    parser.add_argument('--infer_tool', type=str, default='openvino', choices=("openvino", "onnxruntime"), help='选择推理引擎, CPU上openvino更快')
    args = parser.parse_args()

    Datasets = Get_datasets(args.datasets)

    if Datasets:
        infer = SegPipeline(args.model_onnx, Datasets, args.imgsz, args.preprocess_type, args.postprocess_type, args.infer_tool)
        
        t1 = time.time()
        # Read image by OpenCV
        img = cv2.imread(args.source)
        
        # Inference
        infer(img, vis_masks=True)

        print('总时间消耗：{:.3f}s'.format(time.time() - t1))

        




