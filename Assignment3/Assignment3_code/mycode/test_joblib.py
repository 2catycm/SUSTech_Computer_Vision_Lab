#%%
%load_ext autoreload
%autoreload 2
import joblib
from joblib import memory
import time
memory = joblib.Memory('./tmp2', verbose=1)
def a(b):
    time.sleep(3)
    return b+1
a = memory.cache(a)
a(1)
# %%
