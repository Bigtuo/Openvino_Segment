# 利用Openvino和Onnxruntime进行语义分割模型CPU端推理

1 安装

pip install openvino -i  https://pypi.tuna.tsinghua.edu.cn/simple

pip install onnxruntime -i  https://pypi.tuna.tsinghua.edu.cn/simple

其他：opencv-python numpy

2 准备onnx文件，用Netron查看onnx文件, 确定模型输入输出维度,是动态/静态输入,图像输入尺寸

3 确定自己训练数据集, 可在对应文件自定义类别与掩码颜色

4 运行demo_img.py，修改其中对应参数（结果生成outputs中）

注意：支持mmseg训练生成onnx文件
