import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('E:/project_lk/.../yolo11.yaml')
    model.train(data='E:/project_lk/.../data.yaml',
                cache=False,
                imgsz=640,
                epochs=250,
                batch=16,
                # close_mosaic=0, # 最后多少个epoch关闭mosaic数据增强，设置0代表全程开启mosaic训练
                workers=8, # Windows下可以尝试把workers设置为0
                # device='0,1', # 指定显卡和多卡训练
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp | loss出现nan可以关闭amp
                # fraction=0.2,
                project='runs/experiment',
                name='exp',
                )
