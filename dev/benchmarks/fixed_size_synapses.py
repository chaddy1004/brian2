import time
import pickle
from collections import defaultdict
from processify import processify

from joblib import Memory
memory = Memory(location='.', verbose=0)

@processify
@memory.cache
def run_sim(target, n_select, n_total, fixed, trial, method):
    from brian2 import (set_device, device, NeuronGroup, Synapses, run,
                        clear_cache, prefs, second)
    if method == 'tracking':
        prefs.core.algo_cutoff = 1.
    else:
        prefs.core.algo_cutoff = 0.
    assert n_select <= n_total
    if target == 'cpp_standalone':
        set_device('cpp_standalone')
        device.reinit()
        device.activate(directory=None)
    else:
        set_device('runtime')
        if target == 'cython':
            clear_cache(target)

        prefs.codegen.target = target
    source_neurons = NeuronGroup(1000, '')
    target_neurons = NeuronGroup(n_total, '')
    syn = Synapses(source_neurons, target_neurons)
    start = time.time()
    if fixed:
        syn.connect(j='k for k in sample(N_post, size=n_select)')
    else:
        p = n_select / n_total
        syn.connect(j='k for k in sample(N_post, p=p)')
    if target == 'cpp_standalone':
        run(0*second)
    end = time.time()
    if fixed:
        assert len(syn) == n_select*1000
        assert all(syn.N_outgoing_pre == n_select)
    device.delete()
    return end - start


if __name__ == '__main__':
    import numpy as np
    select_p = [0.,  0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
    n_trials = 5
    targets = ['cython', 'cpp_standalone']
    total_size = 100_000
    print(total_size)
    for method in ['tracking', 'selection']:
        print(method)
        results_fixed = defaultdict(list)
        for p in select_p:
            print(p, end=' ')
            select_size = int(p*total_size)
            for target in targets:
                trial_times_fixed = []
                for trial in range(n_trials):
                    t = run_sim(target, select_size, total_size, True, trial,
                                method=method)
                    trial_times_fixed.append(t)
                results_fixed[target].append(np.array(trial_times_fixed))
        print()
        import pprint
        print(total_size)
        pprint.pprint(results_fixed)
        with open(f'synapse_creation_{method}.pickle', 'wb') as f:
            pickle.dump(dict(results_fixed), f)
