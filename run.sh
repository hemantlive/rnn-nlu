data_dir=data/twitter
model_dir=model_tmp
max_sequence_length=50  # max length for train/valid/test sequence
task=joint  # available options: intent; tagging; joint
bidirectional_rnn=True  # available options: True; False
embedding_file=./trained_embedding/twitter_embedding_map.pickle
use_trained_embedding=True
use_manual_edu=False
python3 run_multi-task_rnn.py --data_dir $data_dir \
	--train_dir   $model_dir\
	--max_sequence_length $max_sequence_length \
	--task $task \
	--bidirectional_rnn $bidirectional_rnn \
	--embedding_file $embedding_file \
	--use_trained_embedding $use_trained_embedding \
	--use_manual_edu $use_manual_edu
