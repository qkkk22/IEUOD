from model_wrapper import ConsistencyLossWrapper
from custom_loss import CosineConsistencyLoss

class ConsistencyLossCallback:
    def __init__(self, loss_weight=0.1):
        self.loss_weight = loss_weight
        self.consistency_loss = CosineConsistencyLoss()
        
    def on_train_start(self, trainer):
        """训练开始时包装模型并注册钩子"""
        # 包装原始模型
        trainer.model = ConsistencyLossWrapper(
            trainer.model, 
            loss_weight=self.loss_weight
        )
        
    def on_train_batch_end(self, trainer):
        """训练批次结束后计算并添加一致性损失"""
        if hasattr(trainer.model, 'features') and trainer.model.features is not None:
            # 获取当前批次的原始图像
            # 注意：trainer.train_loader.dataset 应该包含原始图像
            # 实际实现可能需要根据数据加载器调整
            img = trainer.train_loader.dataset.get_current_batch_images()
            
            # 计算一致性损失
            consistency_loss = self.consistency_loss(img, trainer.model.features)
            
            # 添加到总损失
            trainer.loss += self.loss_weight * consistency_loss
            
            # 更新损失项（用于日志记录）
            if hasattr(trainer, 'loss_items'):
                trainer.loss_items = (*trainer.loss_items, consistency_loss.item())
            else:
                trainer.loss_items = (consistency_loss.item(),)
            
            # 清理中间变量
            del trainer.model.features
            trainer.model.features = None
            
    def on_train_end(self, trainer):
        """训练结束时移除钩子"""
        trainer.model.remove_hook()