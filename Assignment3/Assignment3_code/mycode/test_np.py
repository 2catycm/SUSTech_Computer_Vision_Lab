#%%
import numpy as np

#%%
a = np.array([7,100,2,5])

b = np.arange(4).reshape(4,1)

print(a[b])

#%%

a = np.array([
    [3,2,3],
    [4,5,5],
])

from scipy import stats as st
b = st.mode(a,axis=1)
print(b)
print(b.mode)
print(b.count)
print(b.mode.squeeze())
# %%
train_labels = [0,1,2]
k = 1
nearest = np.argsort(a, axis=1)[:, :k].astype(int) # 保留排序索引。 保留前k个最小的。
neighbour_labels = np.array(train_labels)[nearest] # 变成nearest的形状，每个值被train_labels映射。
# %%
print(np.argsort(a, axis=1))
# %%
test_labels = st.mode(neighbour_labels, axis=1).mode.squeeze()
test_labels

# %%
