import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    vi_increases, vi_planning_times, vi_path_lengths = [], [], []
    g_increases, g_planning_times, g_path_lengths = [], [], []
    rand_increases, rand_planning_times, rand_path_lengths = [], [], []

    improvable_initial_q, non_improvable_initial_q = [], []
    with open('/home/chriscorrea14/toppling_simulated1551496920.log', 'r') as file:
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
            if line.startswith('Greedy Path Length: '):
                g_path_length = int(line.split()[-1])
            if line.startswith('Greedy Planning Time: '):
                g_planning_time = float(line.split()[-1])
            if line.startswith('Greedy No Actions: '):
                g_no_actions = eval(line.split()[-1])
            if line.startswith('Greedy Already Best: '):
                g_already_best = eval(line.split()[-1])
            if line.startswith('Random Final Quality: '):
                rand_final_quality = float(line.split()[-1])
            if line.startswith('Random Path Length: '):
                rand_path_length = int(line.split()[-1])
            if line.startswith('Random Planning Time: '):
                rand_planning_time = float(line.split()[-1])

            if line == '\n':
                i += 1
                if (vi_no_actions and g_no_actions) or (vi_already_best and g_already_best):
                    non_improvable_initial_q.append(current_quality)
                    continue
                improvable_initial_q.append(current_quality)
                vi_increases.append(vi_final_quality - current_quality)
                g_increases.append(g_final_quality - current_quality)
                rand_increases.append(rand_final_quality - current_quality)

                vi_planning_times.append(vi_planning_time / (vi_path_length+1))
                g_planning_times.append(g_planning_time / (g_path_length+1))
                rand_planning_times.append(rand_planning_time / (rand_path_length+1))
                
                vi_path_lengths.append(vi_path_length)
                g_path_lengths.append(g_path_length)
                rand_path_lengths.append(rand_path_length)
    print i, len(vi_increases)
    #print vi_increases
    print 'vi increase', np.mean(vi_increases)
    print 'g increase', np.mean(g_increases)
    print 'rand_increase', np.mean(rand_increases)

    print 'vi planning_time', np.mean(vi_planning_times)
    print 'g planning time', np.mean(g_planning_times)
    print 'vi path length', np.mean(vi_path_lengths)
    print 'g path length', np.mean(g_path_lengths)
    print 'rand planning time', np.mean(rand_planning_times)

    plt.hist(non_improvable_initial_q, bins=10)
    plt.title('Initial Grasp Quality for Examples where Toppling is not Useful')
    plt.show()
    plt.hist(improvable_initial_q, bins=10)
    plt.title('Initial Grasp Quality for Examples where Toppling is Useful')
    plt.show()
    plt.hist(vi_increases, bins=10)
    plt.title('Increase in Grasp Quality for the Value Iteration Policy')
    plt.show()

    plt.style.use('seaborn-darkgrid')
    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize=(20,15))
    plt.subplot(1,2,1)
    v_df = pd.DataFrame({'Suction Grasp Quality Increases': vi_increases})
    v_df['Policy Name'] = 'Value Iteration'
    g_df = pd.DataFrame({'Suction Grasp Quality Increases': g_increases})
    g_df['Policy Name'] = 'Greedy'
    r_df = pd.DataFrame({'Suction Grasp Quality Increases': rand_increases})
    r_df['Policy Name'] = 'Random'
    df = pd.concat([r_df, g_df, v_df]) 
    barplot = sns.barplot(x='Policy Name', y='Suction Grasp Quality Increases', data=df, capsize=.2)
    #barplot.set_xticklabels(barplot.get_xticklabels(), rotation=30)
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Suction Grasp Quality Increases')

    plt.subplot(1,2,2)
    v_df = pd.DataFrame({'Planning Time (s)': vi_planning_times})
    v_df['Policy Name'] = 'Value Iteration'
    g_df = pd.DataFrame({'Planning Time (s)': g_planning_times})
    g_df['Policy Name'] = 'Greedy'
    r_df = pd.DataFrame({'Planning Time (s)': rand_planning_times})
    r_df['Policy Name'] = 'Random'
    df = pd.concat([r_df, g_df, v_df])
    barplot = sns.barplot(x='Policy Name', y='Planning Time (s)', data=df, capsize=.2)
    #barplot.set_xticklabels(barplot.get_xticklabels(), rotation=30)
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Planning Time (s)')
    #plt.show()
    plt.savefig('/home/chriscorrea14/simulated_experiments.png')
