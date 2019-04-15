import numpy as np

if __name__ == '__main__':
    useful, all_objs = 0, 0
    with open('/nfs/diskstation/db/toppling/find_objects/dexnet4.log', 'r') as file:
        for line in file:
            if line.startswith('useful'):
                useful += 1
            all_objs += 1
    print useful, all_objs, useful/float(all_objs)
