import numpy as np
import cv2


class Preprocess(object):
    # preprocess_type='Normalization' or 'Standardization'
    def __init__(self, model_width, model_height, preprocess_type='Normalization') -> None:
        self.model_width, self.model_height = model_width, model_height
        self. preprocess_type =  preprocess_type
        pass
    
    # 图片预处理
    def preprocess(self, image, ndtype):
        shape = image.shape[:2]  # original image shape
        img = cv2.resize(image, (self.model_width, self.model_height), interpolation=cv2.INTER_LINEAR)

        # 归一化，直接除255
        if self. preprocess_type == 'Normalization':
            # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
            img = np.ascontiguousarray(np.einsum('HWC->CHW', img)[::-1], dtype=ndtype) / 255.0
            #CHW---> 1,C,H,W
            img = img[None] if len(img.shape) == 3 else img

        # 标准化，首先减去数据的均值，然后除以数据的标准差（mmseg使用）
        else:
            input = img[:,:,::-1].transpose(2,0,1)  #BGR2RGB和HWC2CHW
            input = input.astype(dtype=ndtype)
            input[0,:] = (input[0,:] - 123.675) / 58.395   
            input[1,:] = (input[1,:] - 116.28) / 57.12
            input[2,:] = (input[2,:] - 103.53) / 57.375
            img = np.expand_dims(input, axis=0)

        return img, shape
    

class Postprocess(object):
    # postprocess_type='One' or 'Mutil',表示第二个输出维度是1维还是多维
    def __init__(self, postprocess_type='One') -> None:
        self.postprocess_type = postprocess_type
        pass

    def postprocess(self, preds):
        if self.postprocess_type == 'One':
            predictions = preds[0, 0].astype("uint8")  # (1, 1, 1024, 1024) --> (1024, 1024)

        else:
            # 例如：获取7层矩阵第二维度对应索引的最大值，即当前位置的类别，(1, 7, 1024, 1024) --> (1, 1024, 1024)
            max_index = np.argmax(preds, axis=1)
            predictions = max_index[0].astype("uint8")  # (1, 1024, 1024) --> (1024, 1024) 
        
        return predictions
    


