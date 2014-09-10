import joblib
import subprocess
from config import PYTHON_CMD, CLUSTER_NODES, SSH_CMD, CLUSTER_DIR,\
        CPU_FREE_TOL, RAM_FREE_TOL, NUM_CPUS

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
