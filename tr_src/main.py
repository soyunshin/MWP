# python -m src.main -no-debug -mode train -gpu 1 -dropout 0.1 -heads 4 -encoder_layers 1 -decoder_layers 1 -d_model 768 -d_ff 256 -lr 0.0001 -emb_lr 1e-5 -batch_size 32 -epochs 70 -embedding roberta -emb_name roberta-base -mawps_vocab -dataset mawps_fold0 -run_name mawps_try1
import os
import sys
import math
import logging
import pdb
import random
import numpy as np
from attrdict import AttrDict
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from collections import OrderedDict
try:
	import cPickle as pickle
except ImportError:
	import pickle
import argparse

from args import build_parser
from utils.helper import *
from utils.logger import get_logger, print_log, store_results, store_val_results
from dataloader import TextDataset
from model import build_model, train_model, run_validation

global log_folder
global model_folder
global result_folder
global data_path
global board_path

log_folder = 'C:/Users/soyun/Desktop/svamp_prac/logs' #'logs'
model_folder = 'C:/Users/soyun/Desktop/svamp_prac/models' # 'models'
outputs_folder = 'C:/Users/soyun/Desktop/svamp_prac/outputs' #'outputs'
result_folder = 'C:/Users/soyun/Desktop/svamp_prac/out/' #'./out/'
data_path = 'C:/Users/soyun/Desktop/svamp_prac/data/' #'./data/'
board_path =  'C:/Users/soyun/Desktop/svamp_prac/runs/' #'./runs/'


def load_data(config, logger):
	'''
		Loads the data from the datapath in torch dataset form

		Args:
			config (dict) : configuration/args
			logger (logger) : logger object for logging

		Returns:
			dataloader(s) 
	'''
	
	if config.mode == 'train':
		logger.debug('Loading Training Data...')

		'''Load Datasets'''
		train_set = TextDataset(data_path=data_path, dataset=config.dataset,
								datatype='train', max_length=config.max_length, is_debug=config.debug, is_train=True)
		val_set = TextDataset(data_path=data_path, dataset=config.dataset,
							  datatype='dev', max_length=config.max_length, is_debug=config.debug, grade_info=config.grade_disp, 
							  type_info=config.type_disp, challenge_info=config.challenge_disp)

		'''In case of sort by length, write a different case with shuffle=False '''
		train_dataloader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=5)
		val_dataloader = DataLoader(val_set, batch_size=config.batch_size, shuffle=True, num_workers=5)

		train_size = len(train_dataloader) * config.batch_size
		val_size = len(val_dataloader)* config.batch_size
		
		msg = 'Training and Validation Data Loaded:\nTrain Size: {}\nVal Size: {}'.format(train_size, val_size)
		logger.info(msg)

		return train_dataloader, val_dataloader

	elif config.mode == 'test':
		logger.debug('Loading Test Data...')

		test_set = TextDataset(data_path=data_path, dataset=config.dataset,
							   datatype='test', max_length=config.max_length, is_debug=config.debug)
		test_dataloader = DataLoader(
			test_set, batch_size=config.batch_size, shuffle=True, num_workers=5)

		logger.info('Test Data Loaded...')
		return test_dataloader

	else:
		logger.critical('Invalid Mode Specified')
		raise Exception('{} is not a valid mode'.format(config.mode))


def main(args):
	'''read arguments'''
	# parser = build_parser()
	# args = parser.parse_args()
	config = args
	mode = config.mode
	if mode == 'train':
		is_train = True
	else:
		is_train = False

	''' Set seed for reproducibility'''
	np.random.seed(config.seed)
	torch.manual_seed(config.seed)
	random.seed(config.seed)

	'''GPU initialization'''
	device = gpu_init_pytorch(config.gpu)

	if config.full_cv:
		global data_path 
		data_name = config.dataset
		data_path = data_path + data_name + '/'
		config.val_result_path = os.path.join(result_folder, 'CV_results_{}.json'.format(data_name))
		fold_acc_score = 0.0
		folds_scores = []
		for z in range(5):
			run_name = config.run_name + '_fold' + str(z)
			config.dataset = 'fold' + str(z)
			config.log_path = os.path.join(log_folder, run_name)
			config.model_path = os.path.join(model_folder, run_name)
			config.board_path = os.path.join(board_path, run_name)
			config.outputs_path = os.path.join(outputs_folder, run_name)

			vocab1_path = os.path.join(config.model_path, 'vocab1.p')
			vocab2_path = os.path.join(config.model_path, 'vocab2.p')
			config_file = os.path.join(config.model_path, 'config.p')
			log_file = os.path.join(config.log_path, 'log.txt')

			if config.results:
				config.result_path = os.path.join(result_folder, 'val_results_{}_{}.json'.format(data_name, config.dataset))

			if is_train:
				create_save_directories(config.log_path)
				create_save_directories(config.model_path)
				create_save_directories(config.outputs_path)
			else:
				create_save_directories(config.log_path)
				create_save_directories(config.result_path)

			logger = get_logger(run_name, log_file, logging.DEBUG)
			writer = SummaryWriter(config.board_path)

			logger.debug('Created Relevant Directories')
			logger.info('Experiment Name: {}'.format(config.run_name))

			'''Read Files and create/load Vocab'''
			if is_train:
				train_dataloader, val_dataloader = load_data(config, logger)

				logger.debug('Creating Vocab...')

				voc1 = Voc1()
				voc1.create_vocab_dict(config, train_dataloader)

				# Removed
				# voc1.add_to_vocab_dict(config, val_dataloader)

				voc2 = Voc2(config)
				voc2.create_vocab_dict(config, train_dataloader)

				# Removed
				# voc2.add_to_vocab_dict(config, val_dataloader)

				logger.info('Vocab Created with number of words : {}'.format(voc1.nwords))

				with open(vocab1_path, 'wb') as f:
					pickle.dump(voc1, f, protocol=pickle.HIGHEST_PROTOCOL)
				with open(vocab2_path, 'wb') as f:
					pickle.dump(voc2, f, protocol=pickle.HIGHEST_PROTOCOL)

				logger.info('Vocab saved at {}'.format(vocab1_path))

			else:
				test_dataloader = load_data(config, logger)
				logger.info('Loading Vocab File...')

				with open(vocab1_path, 'rb') as f:
					voc1 = pickle.load(f)
				with open(vocab2_path, 'rb') as f:
					voc2 = pickle.load(f)

				logger.info('Vocab Files loaded from {}\nNumber of Words: {}'.format(vocab1_path, voc1.nwords))

			# TO DO : Load Existing Checkpoints here
			checkpoint = get_latest_checkpoint(config.model_path, logger)

			if is_train:
				model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger)

				logger.info('Initialized Model')

				if checkpoint == None:
					min_val_loss = torch.tensor(float('inf')).item()
					min_train_loss = torch.tensor(float('inf')).item()
					max_val_bleu = 0.0
					max_val_acc = 0.0
					max_train_acc = 0.0
					best_epoch = 0
					epoch_offset = 0
				else:
					epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = \
																		load_checkpoint(model, config.mode, checkpoint, logger, device)

				with open(config_file, 'wb') as f:
					pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

				logger.debug('Config File Saved')

				logger.info('Starting Training Procedure')
				max_val_acc = train_model(model, train_dataloader, val_dataloader, voc1, voc2, device, config, logger, 
							epoch_offset, min_val_loss, max_val_bleu, max_val_acc, min_train_loss, max_train_acc, best_epoch, writer)

			else:
				gpu = config.gpu
				mode = config.mode
				dataset = config.dataset
				batch_size = config.batch_size
				with open(config_file, 'rb') as f:
					config = AttrDict(pickle.load(f))
					config.gpu = gpu
					config.mode = mode
					config.dataset = dataset
					config.batch_size = batch_size

				with open(config_file, 'rb') as f:
					config = AttrDict(pickle.load(f))
					config.gpu = gpu

				model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger)

				epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = \
																		load_checkpoint(model, config.mode, checkpoint, logger, device)

				logger.info('Prediction from')
				od = OrderedDict()
				od['epoch'] = ep_offset
				od['min_train_loss'] = min_train_loss
				od['min_val_loss'] = min_val_loss
				od['max_train_acc'] = max_train_acc
				od['max_val_acc'] = max_val_acc
				od['max_val_bleu'] = max_val_bleu
				od['best_epoch'] = best_epoch
				print_log(logger, od)

				test_acc_epoch = run_validation(config, model, test_dataloader, voc1, voc2, device, logger, 0)
				logger.info('Accuracy: {}'.format(test_acc_epoch))

			fold_acc_score += max_val_acc
			folds_scores.append(max_val_acc)

		fold_acc_score = fold_acc_score/5
		store_val_results(config, fold_acc_score, folds_scores)
		logger.info('Final Val score: {}'.format(fold_acc_score))
			
	else:
		'''Run Config files/paths'''
		run_name = config.run_name
		config.log_path = os.path.join(log_folder, run_name)
		config.model_path = os.path.join(model_folder, run_name)
		config.board_path = os.path.join(board_path, run_name)
		config.outputs_path = os.path.join(outputs_folder, run_name)

		vocab1_path = os.path.join(config.model_path, 'vocab1.p')
		vocab2_path = os.path.join(config.model_path, 'vocab2.p')
		config_file = os.path.join(config.model_path, 'config.p')
		log_file = os.path.join(config.log_path, 'log.txt')

		if config.results:
			config.result_path = os.path.join(result_folder, 'val_results_{}.json'.format(config.dataset))

		if is_train:
			create_save_directories(config.log_path)
			create_save_directories(config.model_path)
			create_save_directories(config.outputs_path)
		else:
			create_save_directories(config.log_path)
			create_save_directories(config.result_path)

		logger = get_logger(run_name, log_file, logging.DEBUG)
		writer = SummaryWriter(config.board_path)

		logger.debug('Created Relevant Directories')
		logger.info('Experiment Name: {}'.format(config.run_name))

		'''Read Files and create/load Vocab'''
		if is_train:
			train_dataloader, val_dataloader = load_data(config, logger)

			logger.debug('Creating Vocab...')

			voc1 = Voc1()
			voc1.create_vocab_dict(config, train_dataloader)

			# Removed
			# voc1.add_to_vocab_dict(config, val_dataloader)

			voc2 = Voc2(config)
			voc2.create_vocab_dict(config, train_dataloader)

			# Removed
			# voc2.add_to_vocab_dict(config, val_dataloader)

			logger.info('Vocab Created with number of words : {}'.format(voc1.nwords))

			with open(vocab1_path, 'wb') as f:
				pickle.dump(voc1, f, protocol=pickle.HIGHEST_PROTOCOL)
			with open(vocab2_path, 'wb') as f:
				pickle.dump(voc2, f, protocol=pickle.HIGHEST_PROTOCOL)

			logger.info('Vocab saved at {}'.format(vocab1_path))

		else:
			test_dataloader = load_data(config, logger)
			logger.info('Loading Vocab File...')

			with open(vocab1_path, 'rb') as f:
				voc1 = pickle.load(f)
			with open(vocab2_path, 'rb') as f:
				voc2 = pickle.load(f)

			logger.info('Vocab Files loaded from {}\nNumber of Words: {}'.format(vocab1_path, voc1.nwords))

		# Load Existing Checkpoints here
		checkpoint = get_latest_checkpoint(config.model_path, logger)

		if is_train:
			model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger)

			logger.info('Initialized Model')

			if checkpoint == None:
				min_val_loss = torch.tensor(float('inf')).item()
				min_train_loss = torch.tensor(float('inf')).item()
				max_val_bleu = 0.0
				max_val_acc = 0.0
				max_train_acc = 0.0
				best_epoch = 0
				epoch_offset = 0
			else:
				epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = \
																	load_checkpoint(model, config.mode, checkpoint, logger, device)

			with open(config_file, 'wb') as f:
				pickle.dump(vars(config), f, protocol=pickle.HIGHEST_PROTOCOL)

			logger.debug('Config File Saved')

			logger.info('Starting Training Procedure')
			train_model(model, train_dataloader, val_dataloader, voc1, voc2, device, config, logger, 
						epoch_offset, min_val_loss, max_val_bleu, max_val_acc, min_train_loss, max_train_acc, best_epoch, writer)

		else:
			gpu = config.gpu
			mode = config.mode
			dataset = config.dataset
			batch_size = config.batch_size
			with open(config_file, 'rb') as f:
				config = AttrDict(pickle.load(f))
				config.gpu = gpu
				config.mode = mode
				config.dataset = dataset
				config.batch_size = batch_size

			with open(config_file, 'rb') as f:
				config = AttrDict(pickle.load(f))
				config.gpu = gpu

			model = build_model(config=config, voc1=voc1, voc2=voc2, device=device, logger=logger)

			epoch_offset, min_train_loss, min_val_loss, max_train_acc, max_val_acc, max_val_bleu, best_epoch, voc1, voc2 = \
																	load_checkpoint(model, config.mode, checkpoint, logger, device)

			logger.info('Prediction from')
			od = OrderedDict()
			od['epoch'] = ep_offset
			od['min_train_loss'] = min_train_loss
			od['min_val_loss'] = min_val_loss
			od['max_train_acc'] = max_train_acc
			od['max_val_acc'] = max_val_acc
			od['max_val_bleu'] = max_val_bleu
			od['best_epoch'] = best_epoch
			print_log(logger, od)

			test_acc_epoch = run_validation(config, model, test_dataloader, voc1, voc2, device, logger, 0)
			logger.info('Accuracy: {}'.format(test_acc_epoch))


args = argparse.Namespace(
  # Mode specifications
  mode = 'train', # choices=['train', 'test', 'conf'], help='Modes: train, test, conf')
  debug = False,

  # Run Config
  run_name ='debug',
  dataset = 'cv_asdiv-a', #default='asdiv-a_fold0_final'
  display_freq = 10000,
  outputs = True,
  # outputs', dest='outputs', action='store_true', help='Show full validation outputs')
  # no-outputs', dest='outputs', action='store_false', help='Do not show full validation outputs')
  # parser.set_defaults(outputs=True)

  results = True,
  # parser.add_argument('-results', dest='results', action='store_true', help='Store results')
  # parser.add_argument('-no-results', dest='results', action='store_false', help='Do not store results')
  # parser.set_defaults(results=True)

  # Meta Attributes
  vocab_size = 30000,
  histogram=True,
  save_writer=False,
  # parser.add_argument('-histogram', dest='histogram', action='store_true', help='Operate in debug mode')
  # parser.add_argument('-no-histogram', dest='histogram', action='store_false', help='Operate in normal mode')
  # parser.set_defaults(histogram=True)
  # parser.add_argument('-save_writer', dest='save_writer',action='store_true', help='To write tensorboard')
  # parser.add_argument('-no-save_writer', dest='save_writer', action='store_false', help='Dont write tensorboard')
  # parser.set_defaults(save_writer=False)

  # Device Configuration
  gpu = 0,  #help='Specify the gpu to use')
  early_stopping = 50,
  seed = 6174,
  logging = 1,
  ckpt = 'model',
  save_model=False,
  # parser.add_argument('-save_model', dest='save_model',action='store_true', help='To save the model')
  # parser.add_argument('-no-save_model', dest='save_model', action='store_false', help='Dont save the model')
  # parser.set_defaults(save_model=False)
  # parser.add_argument('-log_fmt', type=str, default='%(asctime)s | %(levelname)s | %(name)s | %(message)s', help='Specify format of the logger')


	# Transformer parameters
  heads = 4, #8, #help='Number of Attention Heads')
  encoder_layers = 1, #6, #help='Number of layers in encoder')
  decoder_layers = 1, #6, #help='Number of layers in decoder')
  d_model = 768, #help='the number of expected features in the encoder inputs') #768? features of BERT? HAS TO BE 300 if using word2Vec
  d_ff = 256, #help='Embedding dimensions of intermediate FFN Layer (refer Vaswani et. al)')
  lr = 0.0001, #help='Learning rate')
  dropout = 0.1, #help= 'Dropout probability for input/output/state units (0.0: no dropout)')
  warmup = 0.1, #help='Proportion of training to perform linear learning rate warmup for')
  max_grad_norm = 0.25, #help='Clip gradients to this norm')
  batch_size = 16, #help='Batch size')

  max_length =80, #help='Specify max decode steps: Max length string to output')
  init_range =0.08, #help='Initialization range for seq2seq model')
	
  embedding ='electra', #choices=['albert', 'bart' 'bert', 'electra', 'gpt2', 'roberta', 't5', 'xlnet', 'word2vec', 'random'], help='Embeddings')
  word2vec_bin = '/datadrive/satwik/global_data/GoogleNews-vectors-negative300.bin', #help='Binary file of word2vec')
  emb_name ='google/electra-base-discriminator', #choices=['albert-base-v2', 'facebook/bart-base', 'bert-base-uncased', 'google/electra-base-discriminator', 'gpt2', 'roberta-base','t5-base', 'xlnet-base-cased'], help='Which pre-trained model')
  emb_lr =1e-5, #help='Larning rate to train embeddings')
  freeze_emb=False,
	# parser.add_argument('-freeze_emb', dest='freeze_emb', action='store_true', help='Freeze embedding weights')
	# parser.add_argument('-no-freeze_emb', dest='freeze_emb', action='store_false', help='Train embedding weights')
	# parser.set_defaults(freeze_emb=False)

  epochs = 70, #10, #help='Maximum # of training epochs')
  opt ='adamw', #choices=['adam', 'adamw', 'adadelta', 'sgd', 'asgd'], help='Optimizer for training')

  grade_disp=False,
  # parser.add_argument('-grade_disp', dest='grade_disp', action='store_true', help='Display grade information in validation outputs')
  # parser.add_argument('-no-grade_disp', dest='grade_disp', action='store_false', help='Don\'t display grade information')
  # parser.set_defaults(grade_disp=False)
  type_disp=False,
  # parser.add_argument('-type_disp', dest='type_disp', action='store_true', help='Display Type information in validation outputs')
	# parser.add_argument('-no-type_disp', dest='type_disp', action='store_false', help='Don\'t display Type information')
	# parser.set_defaults(type_disp=False)
  challenge_disp=False,
	# parser.add_argument('-challenge_disp', dest='challenge_disp', action='store_true', help='Display information in validation outputs')
	# parser.add_argument('-no-challenge_disp', dest='challenge_disp', action='store_false', help='Don\'t display information')
	# parser.set_defaults(challenge_disp=False)
  nums_disp=True,
	# parser.add_argument('-nums_disp', dest='nums_disp', action='store_true', help='Display number of numbers information in validation outputs')
	# parser.add_argument('-no-nums_disp', dest='nums_disp', action='store_false', help='Don\'t display number of numbers information')
	# parser.set_defaults(nums_disp=True)
  more_nums=False,
	# parser.add_argument('-more_nums', dest='more_nums', action='store_true', help='More numbers in Voc2')
	# parser.add_argument('-no-more_nums', dest='more_nums', action='store_false', help='Usual numbers in Voc2')
	# parser.set_defaults(more_nums=False)
  mawps_vocab= True,#False,
	# parser.add_argument('-mawps_vocab', dest='mawps_vocab', action='store_true', help='Custom Numbers in Voc2')
	# parser.add_argument('-no-mawps_vocab', dest='mawps_vocab', action='store_false', help='No Custom Numbers in Voc2')
	# parser.set_defaults(mawps_vocab=False)

  show_train_acc=True,
	# parser.add_argument('-show_train_acc', dest='show_train_acc', action='store_true', help='Calculate the train accuracy')
	# parser.add_argument('-no-show_train_acc', dest='show_train_acc', action='store_false', help='Don\'t calculate the train accuracy')
	# parser.set_defaults(show_train_acc=False)

  full_cv=True
	# parser.add_argument('-full_cv', dest='full_cv', action='store_true', help='5-fold CV')
	# parser.add_argument('-no-full_cv', dest='full_cv', action='store_false', help='No 5-fold CV')
	# parser.set_defaults(full_cv=False)
  )



if __name__ == '__main__':
	main(args)


''' Just docstring format '''
# class Vehicles(object):
# 	'''
# 	The Vehicle object contains a lot of vehicles

# 	Args:
# 		arg (str): The arg is used for...
# 		*args: The variable arguments are used for...
# 		**kwargs: The keyword arguments are used for...

# 	Attributes:
# 		arg (str): This is where we store arg,
# 	'''
# 	def __init__(self, arg, *args, **kwargs):
# 		self.arg = arg

# 	def cars(self, distance,destination):
# 		'''We can't travel distance in vehicles without fuels, so here is the fuels

# 		Args:
# 			distance (int): The amount of distance traveled
# 			destination (bool): Should the fuels refilled to cover the distance?

# 		Raises:
# 			RuntimeError: Out of fuel

# 		Returns:
# 			cars: A car mileage
# 		'''
# 		pass