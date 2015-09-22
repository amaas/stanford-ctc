# stanford-ctc
Neural net code for lexicon-free speech recognition with connectionist temporal classification

This repository contains code for a bi-directional RNN training using the CTC loss function.
We assume you have separately prepared a dataset of speech utterances with audio features and text transcriptions.

For more information please see the [project page](http://deeplearning.stanford.edu/lexfree/) and the [character language modeling repository](https://github.com/zxie/nn)

Our neural net code runs on the GPU using [Cudamat](https://github.com/cudamat/cudamat)
We use a forked version of Cudamat to add an extra function which you can find [here](https://github.com/awni/cudamat). If you need a more recent version of cudamat you can likely take just the extra function and apply the patch to the most recent version of Cudamat.

The latest code is in the directory `ctc_fast`; please set your `PYTHONPATH` accordingly. The script `runNNet.py` should be the starting point for training the BRNN model -- you'll have to modify `run_cfg.py` and `decoder_config.py`. Unfortunately the `run*.sh` scripts in `{timit/wsj/swbd}-utils` are outdated but you can refer to them for reasonable parameter settings.

Example feat#.bin, keys#.txt, and alis#.txt files for small subset of TIMIT training data can be
found [here](http://deeplearning.stanford.edu/lexfree/timit/).

For details about the algorithms used please see our NAACL paper. Also please cite that paper when using this code:
```
@inproceedings{lexfree2015,
    title={Lexicon-Free Conversational Speech Recognition with Neural Networks},
    author={Maas, Andrew L. and Xie, Ziang and Jurafsky, Dan and Ng, Andrew Y.},
    booktitle={Proceedings the North American Chapter of the Association for Computational Linguistics (NAACL)},
    year={2015}
}
```
