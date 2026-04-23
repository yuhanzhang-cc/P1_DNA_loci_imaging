import pandas as pd

df = pd.read_csv('deepblink.csv')
best_per_image = df.groupby('image')['f1_score'].max()

# 每个图像取最佳 F1 的那一行
best_rows = df.loc[df.groupby('image')['f1_score'].idxmax()]

# 只保留需要的列
best_rows = best_rows[['image', 'f1_score', 'cutoff', 'mean_euclidean']]
best_rows = best_rows.sort_values('image')

print(f"平均最佳 F1: {best_per_image.mean():.3f}")
print(f"标准差: {best_per_image.std():.3f}")
print(f"成功图像数: {(best_per_image > 0).sum()} / {len(best_per_image)}")
print(best_rows.to_string())