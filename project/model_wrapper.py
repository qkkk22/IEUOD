import torch.nn as nn
from custom_loss import CosineConsistencyLoss


class ConsistencyLossWrapper(nn.Module):
    def __init__(self, base_model, loss_weight=0.1):
        """
        Args:
            base_model: 原始YOLO模型
            loss_weight: 一致性损失的权重
        """
        super().__init__()
        self.base_model = base_model
        self.loss_weight = loss_weight
        self.consistency_loss = CosineConsistencyLoss()
        
        # 注册钩子以捕获中间特征
        self.features = None
        self.hook = self._register_hook()
        
    def _register_hook(self):
        """注册前向钩子以捕获中间特征图"""
        # 选择要捕获的中间层（这里以模型的第一个C3层的输出为例）
        target_layer = self._get_target_layer()
        
        # 定义钩子函数
        def hook_fn(module, input, output):
            self.features = output
            
        # 注册钩子
        hook = target_layer.register_forward_hook(hook_fn)
        return hook
        
    def _get_target_layer(self):
        """获取目标层（这里取backbone的最后一个输出层）"""
        # 注意：根据您的模型结构调整
        model = self.base_model.model
        if hasattr(model, 'model'):
            # Ultralytics YOLO模型通常有一个model属性
            model = model.model
            
        # 假设我们取backbone的最后一个输出
        # 在YOLOv8中，backbone通常是model的0到9层
        return model[0]
    
    def forward(self, x):
        """前向传播，返回原始模型的输出"""
        return self.base_model(x)
    
    def remove_hook(self):
        """移除钩子"""
        if self.hook is not None:
            self.hook.remove()