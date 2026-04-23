import numpy as np
from deepblink.io import load_model
from deepblink.inference import predict
from deepblink.metrics import compute_metrics
import pandas as pd

# 加载测试集
data = np.load('dataset157_0415.npz', allow_pickle=True)
x_test = data['x_test']
y_test = data['y_test']
print(f"测试集大小: {len(x_test)} 张图")

# 加载模型
model = load_model('deepblink_dna.h5')
print("模型加载成功")

# 评估
results = []
for i, (img, true) in enumerate(zip(x_test, y_test)):
    pred_raw = predict(img, model, probability=0.5)  # 返回 (N, 3): y, x, p
    pred = pred_raw[:, :2]  # 去掉概率列，只保留 y, x
    metrics = compute_metrics(pred, true, mdist=3)
    results.append(metrics)
    print(f"图 {i+1}: 真实 {len(true)} 点, 预测 {len(pred)} 点")

# 汇总
df = pd.DataFrame(results)
df.to_csv('testset_metrics.csv', index=False)
print("\n=== 测试集汇总 ===")
print(df[['f1_score', 'precision', 'recall', 'rmse']].mean())