import numpy as np

if __name__ == '__main__':
    vi_increases, vi_planning_times, vi_path_lengths, g_increases, g_planning_times, g_path_lengths = [], [], [], [], [], []
    with open('/home/chriscorrea14/toppling_simulated1550747345.log', 'r') as file:
        i = 0
        for line in file:

            if line.startswith('Value Iteration Original Quality: '):
                current_quality = float(line.split()[-1])
            if line.startswith('Value Iteration Final Quality: '):
                vi_final_quality = float(line.split()[-1])
            if line.startswith('Value Iteration Path Length: '):
                vi_path_length = int(line.split()[-1])
            if line.startswith('Value Iteration Planning Time: '):
                vi_planning_time = float(line.split()[-1])
            if line.startswith('Value Iteration No Actions: '):
                vi_no_actions = eval(line.split()[-1])
            if line.startswith('Value Iteration Already Best: '):
                vi_already_best = eval(line.split()[-1])
            if line.startswith('Greedy Final Quality: '):
                g_final_quality = float(line.split()[-1])
            if line.startswith('Greedy Path Pength: '):
                g_path_length = int(line.split()[-1])
            if line.startswith('Greedy Planning Time: '):
                g_planning_time = float(line.split()[-1])
            if line.startswith('Greedy No Actions: '):
                g_no_actions = eval(line.split()[-1])
            if line.startswith('Greedy Already Best: '):
                g_already_best = eval(line.split()[-1])

            if line == '\n':
                i += 1
                if (vi_no_actions and g_no_actions) or (vi_already_best and g_already_best):
                    continue
                vi_increases.append(vi_final_quality - current_quality)
                g_increases.append(g_final_quality - current_quality)
                vi_planning_times.append(vi_planning_time)
                g_planning_times.append(g_planning_time)
                vi_path_lengths.append(vi_path_length)
                g_path_lengths.append(g_path_length)
    print i
    print vi_increases
    print len(vi_increases), np.mean(vi_increases)
    print len(g_increases), np.mean(g_increases)
    print np.mean(vi_planning_times)
    print np.mean(g_planning_times)
    print np.mean(vi_path_lengths)
    print np.mean(g_path_lengths)
