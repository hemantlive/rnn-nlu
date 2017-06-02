Elementary Discourse Unit(EDU) based intent detection and EDU tagging
============
This code is part of NLP (CSS590C) project at University of Washington.
Code modified by: HemantNigam (hnigam@uw.edu)

**Code reused from below paper by Bing Liu Et. Al**

**Dependency **
1. Tensorflow 1.1.0
2. Gensim (latest)
3. Numpy, Scipy (latest)
4. Python 3.5

** Setup **

1. setup SPADE
	a. set CHP var in ./SPADE/SPADE/bin/spade_mod.pl to point bllip-parser 
		e.g. $CHP = <ABSOLUTE_PATH_TO_THIS_REPO>/SPADE/SPADE/parser/bllip-parser/first-stage/PARSE
	b. cd SPADE/SPADE/bin
	c. rm edubreak dependencies
	d. make all

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
