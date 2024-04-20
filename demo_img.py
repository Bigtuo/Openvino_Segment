import cv2
import time
import argparse
from utils.classes import Get_classes
from models.segment import Get_segmenter
from models.seg_pipeline import SegPipeline


if __name__ == '__main__':
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='STDCNet', choices=("STDCNet", "Deeplabv3"), help='分割算法选择')
    parser.add_argument('--model_onnx', type=str, default='weights/tdcnet.onnx', help='Path to ONNX model')
    parser.add_argument('--classes', type=str, default='Custom', choices=("Custom", "Cityscapes"), help='模型训练的类别')
    parser.add_argument('--source', type=str, default=str('img/img.jpg'), help='Path to input image')
    parser.add_argument('--imgsz', type=tuple, default=(1024, 1024), help='Image input size')
    parser.add_argument('--infer_tool', type=str, default='openvino', choices=("openvino", "onnxruntime"), help='选择推理引擎')
    args = parser.parse_args()


    Segmenter = Get_segmenter(args.model_name)
    Classes = Get_classes(args.classes)

    if Segmenter and Classes:
        infer = SegPipeline(Segmenter, args.model_onnx, Classes, args.imgsz, args.infer_tool)
        
        t1 = time.time()
        # Read image by OpenCV
        img = cv2.imread(args.source)
        
        # Inference
        infer(img, vis_masks=True)

        print('总时间消耗：{:.3f}s'.format(time.time() - t1))

        




