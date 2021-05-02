
"""import numpy as np
import time

from numba import vectorize, cuda

@vectorize(['float32(float32, float32)'], target='cuda')
def VectorAdd(a, b):
    return a + b

def main():
    N = 32000000

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    start = time.time()
    C = VectorAdd(A, B)
    vector_add_time = time.time() - start

    print ("C[:5] = " + str(C[:5]))
    print ("C[-5:] = " + str(C[-5:]))

    print ("VectorAdd took for % seconds" % vector_add_time)

if __name__=='__main__':
    main()"""
    

'''from tensorflow.python.client import device_lib
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(get_available_devices())'''

import tensorflow as tf

print(tf.test.is_gpu_available())
print(tf.test.is_built_with_cuda())