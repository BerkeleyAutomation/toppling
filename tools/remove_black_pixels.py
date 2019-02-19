from PIL import Image
import numpy as np
import os
import sys

dir_name = sys.argv[1]

bg_color = np.array([0,0,0])
for fn in os.listdir(dir_name):
    full_fn = os.path.join(dir_name, fn)
    x = Image.open(full_fn)
    x = np.array(x)
    y = np.zeros((x.shape[0], x.shape[1], 4))
    #print x[np.any(x, axis=2)].shape
    mask = np.any(x[:,:,:3], axis=2)
    print mask.shape
    print x.shape, x[mask].shape
    #y[:,:,:3][np.any(x, axis=2)] = x[np.any(x, axis=2)][:,:,:3]
    y[mask] = x[mask]
    print x[500,500], y[500,500]
    y = (255 * y).astype(np.uint8)
    i = Image.fromarray(y)
    i.save(full_fn)
