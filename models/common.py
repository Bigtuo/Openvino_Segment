import numpy as np
from openvino.runtime import Core  # pip install openvino -i  https://pypi.tuna.tsinghua.edu.cn/simple
import onnxruntime as ort  # 使用onnxruntime推理用上，pip install onnxruntime，默认安装CPU


class OpenvinoInference(object):
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        ie = Core()
        self.model_onnx = ie.read_model(model=self.onnx_path)
        self.compiled_model_onnx = ie.compile_model(model=self.model_onnx, device_name="CPU")
        self.output_layer_onnx = self.compiled_model_onnx.output(0)
        self.ndtype = np.single

    def predict(self, datas):
        predict_data = self.compiled_model_onnx([datas])[self.output_layer_onnx]
        return predict_data


class OnnxInference(object):
    def __init__(self, onnx_path):
        self.ort_session = ort.InferenceSession(onnx_path,
                                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                                                if ort.get_device() == 'GPU' else ['CPUExecutionProvider'])
        # Numpy dtype: support both FP32 and FP16 onnx model
        self.ndtype = np.half if self.ort_session.get_inputs()[0].type == 'tensor(float16)' else np.single
    
    def predict(self, datas):
        predict_data = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: datas})[0]
        return predict_data
        