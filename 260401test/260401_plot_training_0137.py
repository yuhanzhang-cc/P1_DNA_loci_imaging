import matplotlib.pyplot as plt

# 从你的输出中提取的数据（每 5 个 epoch 取一次）
epochs = [1, 5, 10, 15, 20]
train_loss = [1.9364, 1.6003, 1.4788, 1.4041, 1.3690]
val_loss = [1.7562, 1.5119, 1.4957, 1.4419, 1.4661]
train_f1 = [0.0045, 0.0057, 0.0021, 0.0019, 0.0019]
val_f1 = [0.0101, 0.0066, 0.0021, 0.0020, 0.0020]

plt.figure(figsize=(12, 4))

# Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b-o', label='Train Loss')
plt.plot(epochs, val_loss, 'r-o', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.grid(True, alpha=0.3)

# F1 曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, train_f1, 'b-o', label='Train F1')
plt.plot(epochs, val_f1, 'r-o', label='Val F1')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.title('F1 Score Curves')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('D:/Projects/P1_DNA_loci_imaging/260401test/training_curves_20260401.png', dpi=150)
print('✅ 曲线图已保存')