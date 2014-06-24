import subprocess
import os
import time
import xml.etree.ElementTree as et


# Available nodes to run on
nodes = [31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]
gorgon = 'gorgon'
tmpScript = 'tmpScript00000000.sh'
base_path = '/scail/group/deeplearning/speech/awni/kaldi-stanford/stanford-nnet/ctc_fast/'

def get_free_gpus(node_name):
    output = subprocess.Popen(['ssh', node_name, 'nvidia-smi','-q','-x'],stdout=subprocess.PIPE).communicate()[0]
    tree = et.fromstring(output.strip())

    gpus = tree.findall('gpu')
    freegpus = [] 
    for i,gpu in enumerate(gpus):
        mem =  gpu.findall('memory_usage')[0]
        tot = int(mem.findall('total')[0].text.split()[0])
        used = int(mem.findall('used')[0].text.split()[0])
        avail = int(mem.findall('free')[0].text.split()[0])
        if float(used)/tot < 0.1:
            freegpus.append(i)

    return freegpus

def get_next_free():
    """
    Loops through node list, returns on first encounter with free gpu.
    Returns node name and device id if free,
    -1,-1 otherwise.
    """
    for n in nodes:
        free_gpus = get_free_gpus(gorgon+str(n))
        if free_gpus:
            return n,free_gpus[0]
    return -1,-1

def write_tmp_script(momentum,layerSize,numLayers,step,
                        anneal,temporalLayer,deviceId):
    script = """#!/bin/bash

momentum=%d
epochs=10
layerSize=%d
numLayers=%d
maxBatch=2000
step=%f
anneal=%f
deviceId=%d
temporalLayer=%d

outfile="%smodels/wsj_layers_${numLayers}_${layerSize}_temporal_${temporalLayer}_step_${step}_mom_${momentum}_anneal_${anneal}.bin"
echo $outfile
python %srunNNet.py --layerSize $layerSize --numLayers $numLayers \
  --step $step --epochs $epochs --momentum $momentum --anneal $anneal \
  --outputDim 34 --inputDim $((41*23)) --rawDim $((41*23))  --numFiles 75 \
  --dataDir /scail/group/deeplearning/speech/awni/kaldi-stanford/kaldi-trunk/egs/wsj/s6/exp/train_si284_ctc/ \
  --maxUttLen $maxBatch --outFile $outfile --deviceId $deviceId --temporalLayer $temporalLayer    
"""%(momentum,layerSize,numLayers,step,anneal,deviceId,temporalLayer,base_path,base_path)

    with open(tmpScript,'w') as fid:
        fid.write(script)
    os.system('chmod +x '+tmpScript)

def remove_tmp_script():
    os.system('rm -f ' + tmpScript)
    

def launch(node,outf,errf):
    script = base_path+tmpScript
    p = subprocess.Popen(['ssh', gorgon+str(node), 'nohup', script], 
            stdout=open(outf,'w'),stderr=open(errf,'w'))

def run():
    momentums = [.9]
    layerSizes = [2048]
    numLayers = [5]
    steps = [1e-5]
    anneals = [1.4]
    temporalLayers = [3]
    for m in momentums:
        for ls in layerSizes:
            for nl in numLayers:
                for s in steps:
                    for a in anneals:
                        for t in temporalLayers:
                            node,deviceId = get_next_free()
                            write_tmp_script(m,ls,nl,s,a,t,deviceId)
                            print "Running job on node %d, device %d"%(node,deviceId)
                            outf=base_path+"models/log/wsj_layers_%d_%d_temporal_%d_step_%f_mom_%f_anneal_%f.out"%(ls,
                                    nl,t,s,m,a)
                            errf=base_path+"models/log/wsj_layers_%d_%d_temporal_%d_step_%f_mom_%f_anneal_%f.err"%(ls,
                                    nl,t,s,m,a)
                            launch(node,outf,errf)
                            time.sleep(30) # give gpu time to get busy before checking next free
    remove_tmp_script()


if __name__=='__main__':
    run()

