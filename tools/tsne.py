import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from toppling import normalize

def compute_gradients(points, values):
    # def grad_helper(points, values):
    #     n_approx = 10
    #     grads = []
    #     for point, value in zip(points, values):
    #         idxes = np.argsort((np.sqrt((points - point)**2).sum(axis=0)))
    #         grad = np.zeros(3)
    #         for idx in idxes[:n_approx]:
    #             grad += (points[idx] - point) * (values[idx] - value)
    #         grads.append(grad / n_approx)
    #     return np.mean(grads, axis=0)
    w1 = normalize(np.linalg.lstsq(points, values)[0])

    projected_points = np.reshape(points.dot(w1), (-1,1)) * np.reshape(w1, (1,-1))
    nullspace_points = points - projected_points
    w2 = normalize(np.linalg.lstsq(nullspace_points, values)[0])
    return w1, w2

def project(points, w1, w2):
    return np.hstack((points.dot(w1).reshape(-1,1), points.dot(w2).reshape(-1,1)))

def plot(params, projected_params, metrics, model, metric_name, lower_better=False):
    # if lower_better:
    #     metrics = 1-metrics
    min_metric, max_metric = np.min(metrics), np.max(metrics)
    metrics = metrics - min_metric
    metrics = metrics / np.max(metrics)
    ann_points = []
    frame1 = plt.gca()
    for i, param, projected, metric in zip(range(len(metrics)), params, projected_params, metrics):
        if lower_better:
            c = [min(1, 2*(metric)), min(2*(1-metric), 1), 0]
        else:
            c = [min(1, 2*(1-metric)), min(2*metric, 1), 0]
        plt.plot(projected[0], projected[1], c=c, marker='o')
        # annotate = True
        # for ann_point in ann_points:
        #     diff = projected - ann_point
        #     diff[1] *= 3
        #     if np.linalg.norm(projected - ann_point) < 5:
        #         annotate = False
        #         break
        if lower_better:
            annotate = metric == np.min(metrics)
        else:
            annotate = metric == np.max(metrics)
        if annotate and len(ann_points) == 0:
            ann_points.append(projected)
            plt.annotate(str(param), xy=projected, xytext=(70, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
            )
    plt.title('{} {} Min: {}, Max: {}'.format(model, metric_name, min_metric, max_metric))
    frame1.axes.get_xaxis().set_ticks([])
    frame1.axes.get_yaxis().set_ticks([])
    plt.show()

if __name__ == '__main__':
    logfile = '/nfs/diskstation/db/toppling/param_sweep_obj_rot.log'
    model_types = ['Baseline', 'Baseline+Rotations', 'Baseline+Robustness', 'Robust Model']
    curr_model = 0
    tsne_results = None
    with open(logfile, 'r') as file:
        params, tvs, l1s, topple_maps, pose_maps = [], [], [], [], []
        for line in file:
            if 'TV CV:' in line:
                params = np.array(params)
                tvs = np.array(tvs)
                print 'best models'
                print params[np.argmin(tvs)], params[np.argmax(pose_maps)], params[np.argmax(topple_maps)]

                # if tsne_results is None:
                #     tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
                #     scaled_params = params - np.mean(params, axis=0)
                #     scaled_params = scaled_params / np.std(scaled_params, axis=0)
                #     tsne_results = tsne.fit_transform(params)

                w1, w2 = compute_gradients(params, topple_maps)
                projected_results = project(params, w1, w2)
                
                # # plot(params, tsne_results, tvs, model_types[curr_model], 'TV', lower_better=True)
                # # plot(params, tsne_results, pose_maps, model_types[curr_model], 'Pose MAP')
                # plot(params, tsne_results, topple_maps, model_types[curr_model], 'Topple MAP')

                # plot(params, projected_results, tvs, model_types[curr_model], 'TV', lower_better=True)
                # plot(params, projected_results, pose_maps, model_types[curr_model], 'Pose MAP')
                plot(params, projected_results, topple_maps, model_types[curr_model], 'Topple MAP')
                params, tvs, l1s, topple_maps, pose_maps = [], [], [], [], []
                curr_model += 1
            if len(line.split('ground friction')) == 1:
                continue
            try:
                ground_friction = float(line.split('ground friction~N(')[-1][:4]) # taking approximately 7 decimal places.  hacky
                finger_friction = float(line.split('finger friction~N(')[-1][:4])
                c_f             = float(line.split('c_f~N(c_f, ')[-1][:7])
                f_f             = float(line.split('f_f~N(f_f, ')[-1][:4])
                r               = float(line.split('R_o~N(I, R_')[-1][:4])
                params.append([ground_friction, finger_friction, c_f, f_f, r])

                tvs.append(float(line.split('TV: ')[-1][:4]))
                pose_maps.append(float(line.split('Pose MAP: ')[-1][:4]))
                topple_maps.append(float(line.split('MAP: ')[-1][:4]))
            except Exception, e:
                print line
                raise e
        
        
