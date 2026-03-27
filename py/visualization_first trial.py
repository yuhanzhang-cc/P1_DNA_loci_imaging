import matplotlib.pyplot as plt
import numpy as np

# 从终端手动记录的数据（你可以在终端里往上翻，记下每 10 个 epoch 的 loss）
epochs = [1, 10, 20, 30, 40, 50]
train_loss = [1.967, 1.750, 1.625, 1.524, 1.527, 1.458]
val_loss = [1.853, 1.641, 1.483, 1.607, 1.738, 1.756]
train_f1 = [0.0019, 0.0017, 0.0015, 0.0015, 0.0016, 0.0016]
val_f1 = [0.0007, 0.0015, 0.0014, 0.0015, 0.0007, 0.0014]

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(epochs, train_loss, 'b-', label='Train Loss')
plt.plot(epochs, val_loss, 'r-', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')

plt.subplot(1,2,2)
plt.plot(epochs, train_f1, 'b-', label='Train F1')
plt.plot(epochs, val_f1, 'r-', label='Val F1')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.title('F1 Score Curves')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print('✅ 曲线图已保存为 training_curves.png')
