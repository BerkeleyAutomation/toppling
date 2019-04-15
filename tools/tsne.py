import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot(params, projected_params, metrics, model, metric_name, lower_better=False):
    if lower_better:
        metrics = 1-metrics
    min_metric, max_metric = np.min(metrics), np.max(metrics)
    metrics = metrics - min_metric
    metrics = metrics / np.max(metrics)
    ann_points = []
    for i, param, projected, metric in zip(range(len(metrics)), params, projected_params, metrics):
        c = [min(1, 2*(1-metric)), min(2*metric, 1), 0]
        plt.plot(projected[0], projected[1], c=c, marker='o')
        annotate = True
        for ann_point in ann_points:
            diff = projected - ann_point
            diff[1] *= 3
            if np.linalg.norm(projected - ann_point) < 4:
                annotate = False
                break
        if annotate:
            ann_points.append(projected)
            plt.annotate(str(param), xy=projected, xytext=(70, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
            )
    plt.title('{} {} Min: {}, Max: {}'.format(model, metric_name, min_metric, max_metric))
    plt.show()

if __name__ == '__main__':
    logfile = '/nfs/diskstation/db/toppling/param_sweep_obj_held_out_500_all_metrics.log'
    model_types = ['Baseline', 'Baseline+Rotations', 'Baseline+Robustness', 'Robust Model']
    curr_model = 0
    tsne_results = None
    with open(logfile, 'r') as file:
        params, tvs, l1s, topple_maps, pose_maps = [], [], [], [], []
        for line in file:
            if 'TV CV:' in line:
                params = np.array(params)
                tvs = np.array(tvs)

                if tsne_results is None:
                    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
                    tsne_results = tsne.fit_transform(params)
                
                plot(params, tsne_results, tvs, model_types[curr_model], 'TV', lower_better=True)
                #plot(params, tsne_results, l1s, model_types[curr_model], 'TV')
                plot(params, tsne_results, pose_maps, model_types[curr_model], 'Pose MAP')
                plot(params, tsne_results, topple_maps, model_types[curr_model], 'Topple MAP')
                params, tvs, l1s, topple_maps, pose_maps = [], [], [], [], []
                curr_model += 1
            if len(line.split('ground friction')) == 1:
                continue
            try:
                ground_friction = float(line.split('ground friction~N(')[-1][:7]) # taking approximately 7 decimal places.  hacky
                finger_friction = float(line.split('finger friction~N(')[-1][:7])
                c_f             = float(line.split('c_f~N(c_f, ')[-1][:7])
                f_f             = float(line.split('f_f~N(f_f, ')[-1][:7])
                params.append([ground_friction, finger_friction, c_f, f_f])

                tvs.append(float(line.split('TV: ')[-1][:4]))
                #l1s.append(float(line.split('L1: ')[-1][:7]))
                pose_maps.append(float(line.split('Pose MAP: ')[-1][:4]))
                topple_maps.append(float(line.split('MAP: ')[-1][:4]))
            except Exception, e:
                print line
                raise e
        
        
