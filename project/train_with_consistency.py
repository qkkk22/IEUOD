from ultralytics import YOLO
from callbacks import ConsistencyLossCallback

def main():
    # 加载模型
    model = YOLO('model/yolov12n.pt')
    
    # 创建回调实例
    consistency_callback = ConsistencyLossCallback(loss_weight=0.1)
    
    # 添加回调
    model.add_callback("on_train_start", consistency_callback.on_train_start)
    model.add_callback("on_train_batch_end", consistency_callback.on_train_batch_end)
    model.add_callback("on_train_end", consistency_callback.on_train_end)
    
    # 如果需要，替换数据加载器
    # original_dataset = ... # 获取原始数据集
    # custom_dataset = CustomDataset(original_dataset)
    # model.train_loader = torch.utils.data.DataLoader(custom_dataset, ...)
    
    # 开始训练
    model.train(
        data='data/dehaze.yaml',
        epochs=100,
        imgsz=640,
        batch=16
    )

if __name__ == '__main__':
    main()