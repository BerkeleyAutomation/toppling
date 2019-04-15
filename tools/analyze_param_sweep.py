import numpy as np

if __name__ == '__main__':
    with open('/nfs/diskstation/db/toppling/param_sweep.log', 'r') as file:
        lines, tvs = [], []
        for i, line in enumerate(file):
            # if i > 126 or not line.startswith('ground friction'):
            if i < 126 or not line.startswith('ground friction'):
                continue
            combined_tv = line.split('Combined TV: ')[1]
            combined_tv = combined_tv.split(' ')[0]
            lines.append(line)
            tvs.append(combined_tv)
        best_combined_tvs = np.argsort(tvs)[:10]
        print best_combined_tvs
        for i in best_combined_tvs:
            print lines[i]
