Elementary Discourse Unit(EDU) based intent detection and EDU tagging
============
This code is part of NLP (CSS590C) project at University of Washington.
Author: HemantNigam (hnigam@uw.edu)

**Code reused from below paper by Bing Liu Et. Al**

** Top level Dependency **
1. Tensorflow 1.1.0
2. Gensim (latest)
3. Numpy, Scipy (latest)
4. Python 3.5

** Recommended System **
1. Cuda enabled GPU with minimum 8GB
2. Linux/Unix OS with root access

** Setup **

1. setup SPADE
	a. set CHP var in ./SPADE/SPADE/bin/spade_mod.pl to point bllip-parser 
		e.g. $CHP = <ABSOLUTE_PATH_TO_THIS_REPO>/SPADE/SPADE/parser/bllip-parser/first-stage/PARSE
	b. cd SPADE/SPADE/bin
	c. rm edubreak dependencies
	d. make all


** Usage **
1. ./run.sh - for edu level embeddings
2. ./run_manual.sh - for edu level embeddings (manually segmented)
3. ./run_word.sh - for word level embeddings (baseline)

Modifiable option (in run.sh and run_manual.sh only): 
1. task: tested option : joint,intent  
2. use_trained_embedding : True = EDU level embedding trained using twitter vocabulary, False = EDU level embedding using random embedding (using tensorflow api)



** References **
1. SPADE
	a. Paper :  Sentence Level Discourse Parsing using Syntactic and Lexical Information by Radu Soricut and Daniel Marcu
	b. url : http://www.isi.edu/licensed-sw/spade/
2.Sentece2Vec
	a. Paper : Distributed Representations of Sentences and Documents by Quoc Le and Tomas Mikolov
	b. Github url : https://github.com/klb3713/sentence2vec
3. BLLIP reranking parser (Charniak Parser)
	a. https://github.com/BLLIP/bllip-parser



Attention-based RNN model for Spoken Language Understanding (Intent Detection & Slot Filling)
==================

Tensorflow implementation of attention-based LSTM models for sequence classification and sequence labeling.

**Setup**

* Tensorflow, version >= r0.9 (https://www.tensorflow.org/versions/r0.9/get_started/index.html)

**Usage**:
```bash
data_dir=data/ATIS_samples
model_dir=model_tmp
max_sequence_length=50  # max length for train/valid/test sequence
task=joint  # available options: intent; tagging; joint
bidirectional_rnn=True  # available options: True; False

python run_multi-task_rnn.py --data_dir $data_dir \
      --train_dir   $model_dir\
      --max_sequence_length $max_sequence_length \
      --task $task \
      --bidirectional_rnn $bidirectional_rnn
```

**Reference**

* Bing Liu, Ian Lane, "Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling", Interspeech, 2016 (<a href="http://www.isca-speech.org/archive/Interspeech_2016/pdfs/1352.PDF" target="_blank">PDF</a>)

```
@inproceedings{Liu+2016,
author={Bing Liu and Ian Lane},
title={Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling},
year=2016,
booktitle={Interspeech 2016},
doi={10.21437/Interspeech.2016-1352},
url={http://dx.doi.org/10.21437/Interspeech.2016-1352},
pages={685--689}
}
```

**Contact** 

Feel free to email liubing@cmu.edu for any pertinent questions/bugs regarding the code. 
