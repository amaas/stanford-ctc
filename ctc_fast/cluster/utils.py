import xml
import joblib
import subprocess
import xml.etree.ElementTree as et
from config import PYTHON_CMD, CLUSTER_NODES, SSH_CMD, CLUSTER_DIR,\
        CPU_FREE_TOL, RAM_FREE_TOL, NUM_CPUS, GPU_FREE_TOL

def is_node_free(node_name):
    print node_name
    try:
        cpu_usage = float(subprocess.Popen(SSH_CMD.split(' ') + [node_name,
            PYTHON_CMD, '%s/scripts/get_cpu_usage.py' % CLUSTER_DIR],
            stdout=subprocess.PIPE).communicate()[0])
        cpu_usage /= 100.0
        mem_usage = float(subprocess.Popen(SSH_CMD.split(' ') + [node_name,
            PYTHON_CMD, '%s/scripts/get_mem_usage.py' % CLUSTER_DIR],
            stdout=subprocess.PIPE).communicate()[0])
    except Exception as e:
        print e
        return False
    print cpu_usage, mem_usage
    return cpu_usage < CPU_FREE_TOL and mem_usage < RAM_FREE_TOL


def get_free_nodes(cluster, parallel=True):
    nodes = [cluster + str(node) for node in CLUSTER_NODES[cluster]]
    if parallel:
        is_free = joblib.Parallel(n_jobs=NUM_CPUS)(
            joblib.delayed(is_node_free)(node) for node in nodes)
    else:
        is_free = list()
        for node in nodes:
            is_free.append(is_node_free(node))
    free_nodes = [nodes[k] for k in range(len(nodes)) if is_free[k]]
    return free_nodes


def run_cpu_job(node, cmd, stdout=subprocess.PIPE, blocking=True):
    cmd = '%s %s "%s"' % (SSH_CMD, node, cmd)
    print 'Running on %s' % node
    print cmd
    if not blocking:
        subprocess.Popen(cmd, stdout=stdout, shell=True)
    else:
        output = subprocess.Popen(cmd,
                stdout=stdout, shell=True).communicate()[0]
        return output


def get_free_gpus(node_name):
    '''
    Based off Awni's runAll.py
    '''
    output = subprocess.Popen(SSH_CMD.split(' ') + [node_name, 'nvidia-smi', '-q', '-x'],
            stdout=subprocess.PIPE).communicate()[0]
    if not output:
        print 'No output for %s' % node_name
        return []
    try:
        tree = et.fromstring(output.strip())
    except xml.parsers.expat.ExpatError:
        print 'Invalid XML: ', output.strip()
        return []

    gpus = tree.findall('gpu')
    print 'Detected %d gpus on %s' % (len(gpus), node_name)
    free_gpus = []
    for i, gpu in enumerate(gpus):
        mem = gpu.findall('memory_usage')
        if len(mem) == 0:
            mem = gpu.findall('fb_memory_usage')
        if len(mem) == 0:
            print 'Couldn\'t get memory usage on %s' % node_name
            return []
        mem = mem[0]
        tot = int(mem.findall('total')[0].text.split()[0])
        used = int(mem.findall('used')[0].text.split()[0])
        print used, '/', tot
        if float(used) / tot < GPU_FREE_TOL:
            free_gpus.append(i)

    return free_gpus


def get_all_free_gpus(cluster, parallel=True):
    nodes = [cluster + str(node) for node in CLUSTER_NODES[cluster]]
    if parallel:
        free_gpus = joblib.Parallel(n_jobs=NUM_CPUS)(
            joblib.delayed(get_free_gpus)(node) for node in nodes)
    else:
        free_gpus = list()
        for node in nodes:
            free_gpus.append(get_free_gpus(node))
    free_gpus_map = dict()
    for fg, n in zip(free_gpus, nodes):
        if len(fg) > 0:
            free_gpus_map[n] = fg
    return free_gpus_map


# Good for when other jobs use multiple GPUs on the same machine
def get_next_free_gpu_sequential(all_free_gpus, used_gpus, flagged_gpus):
    keys = sorted(all_free_gpus.keys())
    for key in keys:
        for gpu in all_free_gpus[key]:
            gpu_name = key + '_' + str(gpu)
            if gpu_name not in used_gpus and gpu_name not in flagged_gpus:
                return key, gpu
    return None, None


def run_gpu_job(node, gpu, cmd, blocking=True, stdout=None):
    # TODO Need to ensure that the run folder names don't conflict
    # for training jobs
    print 'Running on %s gpu %d' % (node, gpu)
    if not blocking:
        cmd = '%s %s "export CUDA_DEVICE=%d; %s"' % (SSH_CMD, node, gpu, cmd)
        print cmd
        subprocess.Popen(cmd,
                stdout=stdout, stderr=subprocess.STDOUT, shell=True)
    else:
        cmd = '%s %s "export CUDA_DEVICE=%d; %s"' % (SSH_CMD, node, gpu, cmd)
        print cmd
        if not stdout:
            stdout = subprocess.PIPE
        output = subprocess.Popen(cmd,
                stdout=stdout, shell=True).communicate()[0]
        return output
