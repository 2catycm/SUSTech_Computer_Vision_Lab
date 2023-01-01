from functools import lru_cache
import numpy as np
class A:
    def __init__(self, array):
      self.array = array
      
@lru_cache(maxsize=4)
def fib(array, n):
    return array.shape+n
    
array = np.random.rand(10, 10)
a = A(array)
print(fib(a.array, 1))