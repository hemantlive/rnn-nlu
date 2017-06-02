# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 09:33:32 2016

@author: Bing Liu (liubing@cmu.edu)

Prepare data for multi-task RNN model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import logging

from tensorflow.python.platform import gfile
import sys
sys.path.insert(0, './sentence2vec')
#print(sys.path)
import os
from sentence2vec import word2vec as w2v
#Word2Vec, Sent2Vec, LineSentence
import csv
import pickle
import subprocess
import numpy as np

spade_output_delimiter = "<S>"
spade_prefix = "<s> "
spade_suffix = " </s>"


# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _UNK]

START_VOCAB_dict = dict()
START_VOCAB_dict['with_padding'] = [_PAD, _UNK]
START_VOCAB_dict['no_padding'] = [_UNK]


PAD_ID = 0

UNK_ID_dict = dict()
UNK_ID_dict['with_padding'] = 1
UNK_ID_dict['no_padding'] = 0

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile(r"\d")

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

def naive_tokenizer(sentence):
  """Naive tokenizer: split the sentence by space into a list of tokens."""
  return sentence.split()  


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = START_VOCAB_dict['with_padding'] + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    #print(vocab)
    #print(rev_vocab)
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, UNK_ID,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: a string, the sentence to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """
  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True, use_padding=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="r") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          if use_padding:
            UNK_ID = UNK_ID_dict['with_padding']
          else:
            UNK_ID = UNK_ID_dict['no_padding']
          token_ids = sentence_to_token_ids(line, vocab, UNK_ID, tokenizer,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")



def create_label_vocab(vocabulary_path, data_path):
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        label = line.strip()
        vocab[label] = 1
      label_list = START_VOCAB_dict['no_padding'] + sorted(vocab)
      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for k in label_list:
          vocab_file.write(k + "\n")

def prepare_multi_task_data(data_dir, in_vocab_size, out_vocab_size,manual_edu):
    train_path = data_dir + '/train/train'
    dev_path = data_dir + '/dev/dev'
    test_path = data_dir + '/test/test'
    
     
    # Create vocabularies of the appropriate sizes.
    in_vocab_path = os.path.join(data_dir, "in_vocab_%d.txt" % in_vocab_size)
    out_vocab_path = os.path.join(data_dir, "out_vocab_%d.txt" % out_vocab_size)
    label_path = os.path.join(data_dir, "label.txt")

    #'''  
    logging.info("Preparing training data")
    if manual_edu == False:
        convert_text2spade_format(data_dir + "/twitter.raw", data_dir + "/twitter.spade.in")
        perform_discourse(data_dir + "/twitter.spade.in", data_dir + "/twitter.spade.seg.out", data_dir + "/twitter.spade.par.out")
    
    source_list = convert_edu2list(data_dir+"/twitter.spade.seg.out",train_path+".edu")
     
    #create edu vocab for all data
    create_edu_vocabulary(in_vocab_path, source_list, in_vocab_size)
    # edu_label and label -
    create_vocabulary(out_vocab_path, train_path + ".edu.label.txt", out_vocab_size, tokenizer=naive_tokenizer)
    create_label_vocab(label_path, train_path + ".label.txt")
    #'''
    # Create token ids for the training data.
    in_seq_train_ids_path = train_path + (".ids%d.seq.in" % in_vocab_size)
    out_seq_train_ids_path = train_path + (".ids%d.seq.out" % out_vocab_size)
    label_train_ids_path = train_path + (".ids.label")
    #'''
    data_to_token_ids_edu(source_list, in_seq_train_ids_path, in_vocab_path)
    data_to_token_ids(train_path + ".edu.label.txt", out_seq_train_ids_path, out_vocab_path, tokenizer=naive_tokenizer)
    data_to_token_ids(train_path + ".label.txt", label_train_ids_path, label_path, normalize_digits=False, use_padding=False)
    
    # Create token ids for the development data.
    logging.info("Preparing dev data")
    if not manual_edu:
        convert_text2spade_format(dev_path + ".raw", dev_path + ".spade.in")
        perform_discourse(dev_path + ".spade.in", dev_path + ".spade.seg.out", dev_path + ".spade.par.out")
    source_list = convert_edu2list(dev_path+".spade.seg.out",dev_path+".edu")
    #''' 
    
    in_seq_dev_ids_path = dev_path + (".ids%d.seq.in" % in_vocab_size)
    out_seq_dev_ids_path = dev_path + (".ids%d.seq.out" % out_vocab_size)
    label_dev_ids_path = dev_path + (".ids.label")
    #'''
    data_to_token_ids_edu(source_list, in_seq_dev_ids_path, in_vocab_path)
    data_to_token_ids(dev_path + ".edu.label.txt", out_seq_dev_ids_path, out_vocab_path, tokenizer=naive_tokenizer)
    data_to_token_ids(dev_path + ".label.txt", label_dev_ids_path, label_path, normalize_digits=False, use_padding=False)
    
    # Create token ids for the test data.
    logging.info("Preparing test data")
    if not manual_edu:
        convert_text2spade_format(test_path + ".raw", test_path + ".spade.in")
        perform_discourse(test_path + ".spade.in", test_path + ".spade.seg.out", test_path + ".spade.par.out")
    source_list = convert_edu2list(test_path+".spade.seg.out",test_path+".edu")
    #'''
    in_seq_test_ids_path = test_path + (".ids%d.seq.in" % in_vocab_size)
    out_seq_test_ids_path = test_path + (".ids%d.seq.out" % out_vocab_size)
    label_test_ids_path = test_path + (".ids.label")
    #'''
    data_to_token_ids_edu(source_list, in_seq_test_ids_path, in_vocab_path)
    data_to_token_ids(test_path + ".edu.label.txt", out_seq_test_ids_path, out_vocab_path, tokenizer=naive_tokenizer)
    data_to_token_ids(test_path + ".label.txt", label_test_ids_path, label_path, normalize_digits=False, use_padding=False)
    #'''


    return (in_seq_train_ids_path, out_seq_train_ids_path, label_train_ids_path,
          in_seq_dev_ids_path, out_seq_dev_ids_path, label_dev_ids_path,
          in_seq_test_ids_path, out_seq_test_ids_path, label_test_ids_path,
          in_vocab_path, out_vocab_path, label_path)

def create_edu_vocabulary(vocabulary_path, text_list, max_vocabulary_size):
  if not gfile.Exists(vocabulary_path):
      print("Creating vocabulary %s from data" % (vocabulary_path))
      vocab = {}
      counter = 0
      for edu_list in text_list:
          counter += 1
          if counter % 100000 == 0:
              print("  processing line %d" % counter)
          for edu in edu_list:
              #print(edu)
              if edu in vocab:
                  vocab[edu] += 1
              else:
                  vocab[edu] = 1
      vocab_list = START_VOCAB_dict['with_padding'] + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
          vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
          for w in vocab_list:
              vocab_file.write(w + "\n")

def initialize_vocabulary_edu(vocabulary_path):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        #print(rev_vocab)
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        #print(vocab)
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids_edu(sentence, vocabulary, UNK_ID):
    return [vocabulary.get(edu.strip(), UNK_ID) for edu in sentence]

def data_to_token_ids_edu(text_list, target_path, vocabulary_path,use_padding=True):
    if not gfile.Exists(target_path):
        print("Tokenizing data in ")
    vocab, _ = initialize_vocabulary_edu(vocabulary_path)
    with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in text_list:
            counter += 1
            if counter % 100000 == 0:
                print("  tokenizing line %d" % counter)
            if use_padding:
                UNK_ID = UNK_ID_dict['with_padding']
            else:
                UNK_ID = UNK_ID_dict['no_padding']
            token_ids = sentence_to_token_ids_edu(line, vocab, UNK_ID)
            tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

"""
    Converts SPADE output to edu_list and saves it to file
    @raw_file = spade segmentation putput file
    @edu
"""
def convert_edu2list(raw_file,edu_file):
    text_list = []

    with open(raw_file, "r") as rf:
        with open(edu_file,"w") as ef:
            newSentence = True
            array = []
            for line in rf:
                line = line.rstrip('\n')
                if line.endswith(spade_output_delimiter):
                    array.append(line[:-3])
                    ef.write(line[:-3].join("\n"))
                    newSentence = True
                else:
                    array.append(line)
                    ef.write(line.join("\n"))
                    newSentence = False
                if newSentence:
                    text_list.append(array)
                    array = []
    #print text_list
    #print len(text_list)
    return text_list        

"""
   Converts raw data to SPADE format
   @input_file = file containing raw data, each line corresponding to new post/chat
   @output_file = data in spade format
                  "raw_data" ----> "<s> raw_data </s>"
"""

def convert_text2spade_format(input_file,output_file):
    with open(input_file, 'r') as inf:
        with open(output_file, 'w') as of:
            for line in inf:
                of.write('%s%s%s\n' % (spade_prefix, line.rstrip('\n'), spade_suffix))

"""
    save_edu_vec
    @orig_file =
    @spade_output_file
    @edu_file = formatted spade_output_file

    
"""
def save_edu_vec(raw_file, spade_out_seg_file, edu_file, pickle_file, map_pickle_file, vocabulary, vocab_size, embedding_dim,use_padding=True):
    edu_list_list = convert_edu2list(spade_out_seg_file, edu_file)
    model = w2v.Word2Vec(w2v.LineSentence(raw_file), size=128, window=5, sg=0, min_count=5, workers=8)
    model.save(raw_file + '.model')
    model.save_word2vec_format(raw_file + '.vec')
    model = w2v.Sent2Vec(w2v.LineSentence(edu_file), model_file=raw_file + '.model')
    sents = model.save_sent2vec_format(edu_file + '.vec')
    initW = np.random.uniform(-0.25,0.25,(vocab_size, embedding_dim))
    idx = 0
    edu_vec_list = []
    if use_padding:
        UNK_ID = UNK_ID_dict['with_padding']
    else:
        UNK_ID = UNK_ID_dict['no_padding']

    for edu_list in edu_list_list:
        vec_list = []
        for edu in edu_list:
            map_idx = vocabulary.get(edu.strip(), UNK_ID)
            #print(map_idx)
            initW[map_idx] = sents[idx]
            vec_list.append(sents[idx])
            idx = idx+1
        edu_vec_list.append(vec_list)

    #print(idx)
    #print len(edu_vec_list)
    #print (edu_vec_list)
    with open(pickle_file, "wb") as fp:
        pickle.dump(edu_vec_list, fp)
    with open(map_pickle_file, "wb") as fp:
        pickle.dump(initW, fp)

def load_edu_vec(data_path):
    with open(data_path + "_source.pickle", "rb") as fp:
        edu_vec_list = pickle.load(fp)
    return edu_vec_list

def load_edu_label(data_path):
    with open(data_path + "_target.pickle", "rb") as fp:
        edu_label_list = pickle.load(fp)
    return edu_label_list

def load_label(data_path):
    with open(data_path + "_label.pickle", "rb") as fp:
        label_list = pickle.load(fp)
    return label_list

def save_label(label_file, pickle_file):
    label_list = []
    with open(label_file, 'r') as f:
        reader = csv.reader(f, dialect='excel', delimiter='\t')
        for row in reader:
            label_list.append(row)
    #print label_list
    with open(pickle_file, "wb") as fp:
        pickle.dump(label_list, fp)

def save_edu_label(edu_label_file, pickle_file):
    edu_label_list = []
    with open(edu_label_file, 'r') as f:
        reader = csv.reader(f, dialect='excel', delimiter='\t')
        for row in reader:
            edu_label_list.append(row)
    #print edu_label_list
    with open(pickle_file, "wb") as fp:
        pickle.dump(edu_label_list, fp)

def prepare_data(data_dir):
    train_path = data_dir + '/train/train'
    dev_path = data_dir + '/dev/dev'
    test_path = data_dir + '/test/test'

    logging.info("Preparing training data")
    convert_text2spade_format(train_path + ".raw", train_path + ".spade.in")
    perform_discourse(train_path + ".spade.in", train_path + ".spade.seg.out", train_path + ".spade.par.out")
    #save_edu_vec(train_path + ".raw", train_path + ".spade.seg.out", train_path + ".edu", train_path + "_source.pickle")
    save_label(train_path + ".label", train_path + "_label.pickle")
    save_edu_label(train_path + ".edu.label", train_path + "_target.pickle")
    
    logging.info("Preparing dev data")
    convert_text2spade_format(dev_path + ".raw", dev_path + ".spade.in")
    perform_discourse(dev_path + ".spade.in", dev_path + ".spade.seg.out", dev_path + ".spade.par.out")
    #save_edu_vec(dev_path + ".raw", dev_path + ".spade.seg.out", dev_path + ".edu", dev_path + "_source.pickle")
    save_label(dev_path + ".label", dev_path + "_label.pickle")
    save_edu_label(dev_path + ".edu.label", dev_path + "_target.pickle")
    
    logging.info("Preparing test data")
    convert_text2spade_format(test_path + ".raw", test_path + ".spade.in")
    perform_discourse(test_path + ".spade.in", test_path + ".spade.seg.out", test_path + ".spade.par.out")
    #save_edu_vec(test_path + ".raw", test_path + ".spade.seg.out", test_path + ".edu", test_path + "_source.pickle")
    save_label(test_path + ".label", test_path + "_label.pickle")
    save_edu_label(test_path + ".edu.label", test_path + "_target.pickle")

def perform_discourse(input_file, seg_output_file, par_output_file):
    abs_seg_file_path = os.path.abspath(seg_output_file)
    abs_par_file_path = os.path.abspath(par_output_file)
    abs_inp_file_path = os.path.abspath(input_file)
    print(abs_seg_file_path)
    print(abs_par_file_path)
    print(abs_inp_file_path)
    command = "cd ./SPADE/SPADE/bin; perl ./spade_mod.pl"
    command_seg = command + " -seg-only " + abs_inp_file_path + " >& " + abs_seg_file_path + "; cd -;"
    command_par = command + " " + abs_inp_file_path + " >& " + abs_par_file_path + "; cd -;"

    logging.info("Performing discourse parsing")
    subprocess.call(["sh", "-c", command_par])
    
    logging.info("Performing discourse segmentation")
    ret_code = subprocess.call(["sh", "-c", command_seg])


'''
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s" % " ".join(sys.argv))
    prepare_data("./data/twitter") 
    #convert_edu2list("output","twitter.edu")
    #convert_text2spade_format("twitter.orig", "twitter.spade")
    #save_edulist2vec("twitter.orig", "output", "twitter.edu")
    #load_edu_label("edu_label.txt")
    program = os.path.basename(sys.argv[0])
    logging.info("finished running %s" % program)
'''

def prepare_edu_embeddings(data_path, vocabulary, vocab_size, embedding_dim,use_padding,manual_edu): 
    logging.info("Preparing word embeddings")
    if not manual_edu:
        convert_text2spade_format(data_path + "/twitter.raw", data_path + "/twitter.spade.in")
        perform_discourse(data_path+ "/twitter.spade.in", data_path + "/twitter.spade.seg.out", data_path + "/twitter.spade.par.out")
    save_edu_vec(data_path + "/twitter.raw", data_path + "/twitter.spade.seg.out", data_path + "/twitter.edu", data_path + "/twitter_source.pickle", data_path + "/twitter_embedding_map.pickle", vocabulary, vocab_size, embedding_dim,use_padding)
    
