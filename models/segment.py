import numpy as np


def Get_segmenter(class_name):
    # 使用 globals() 获取类  
    cls = globals().get(class_name)  
    
    # 确保我们找到了一个类（而不是 None 或其他非类对象）  
    if cls is not None and issubclass(cls, object):  
        # 创建类的实例  
        instance = cls()  
        return instance
    else:  
        print(f"分割算法： {class_name} not found， 名字不对或不存在！")
        return False
    

class STDCNet():
    '''
    https://github.com/MichaelFan01/STDC-Seg?tab=readme-ov-file
    '''
    def __init__(self) -> None:
        pass

    def postprocess(self, preds):
        predictions = preds[0, 0]  # (1, 1, 1024, 1024) --> (1024, 1024)
        
        return predictions


class Deeplabv3():
    def __init__(self) -> None:
        pass

    def postprocess(self, preds):
        # 获取7层矩阵第二维度对应索引的最大值，即当前位置的类别，(1, 7, 1024, 1024) --> (1, 1024, 1024)
        max_index = np.argmax(preds, axis=1)
        predictions = max_index[0].astype("uint8")  # (1, 1024, 1024) --> (1024, 1024)
        return predictions