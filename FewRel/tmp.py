from fewshot_re_kit.data_loader import get_loader, get_loader_pair, get_loader_unsupervised
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import CNNSentenceEncoder, BERTSentenceEncoder, BERTPAIRSentenceEncoder, RobertaSentenceEncoder, RobertaPAIRSentenceEncoder
import models
from models.proto import Proto

import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os


		

def main():
	trainN = 10
	N = 5
	K = 5
	Q = 5
	batch_size = 4
	model_name = 'proto'
	encoder_name = 'cnn'
	optimizer = 'adam'
	max_length = 128
	train = 'train_wiki'
	val = 'val_semeval'
	test = 'val_pubmed'
	train_iter = 30000
	val_iter = 1000
	test_iter = 10000
	val_step = 2000
	lr = 0.1
	weight_decay = 0.00001
	dropout = 0
	na_rate = 0
	grad_iter = 1
	hidden_size = 230
	load_ckpt = None
	save_ckpt = None
	fp16 = False
	only_test = False
	pair = False
	pretrain_ckpt = None
	
	print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
	print("model:{}".format(model_name))
	print("encoder:{}".format(encoder_name))
	print("max_length:{}".format(max_length))
	
	if encoder_name == 'cnn':
		try:
			glove_mat = np.load('./pretrain/glove/glove_mat.npy')
			glove_word2id = json.load(open('./pretrain/glove/glove_word2id.json'))
		except:
			raise Exception("Cannot find glove files. Run glove/download_glove.sh to download glove files.")
		sentence_encoder = CNNSentenceEncoder(
			glove_mat,
			glove_word2id,
			max_length
		)
	else:
		raise NotImplementedError
	
	train_data_loader = get_loader(train, sentence_encoder,
	        N = trainN, K = K, Q = Q, na_rate = na_rate, batch_size = batch_size)
	val_data_loader = get_loader(val, sentence_encoder,N = N, K = K, Q = Q, na_rate = na_rate, batch_size = batch_size)
	test_data_loader = get_loader(test, sentence_encoder,N = N, K = K, Q = Q, na_rate = na_rate, batch_size = batch_size)

	if optimizer == 'sgd':
		pytorch_optim = optim.SGD
	elif optimizer == 'adam':
		pytorch_optim = optim.Adam
	elif optimizer == 'bert_adam':
		from pytorch_transformers import AdamW
		pytorch_optim = AdamW
	else:
		raise NotImplementedError
	
	framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
		
	prefix = '-'.join([model_name, encoder_name, train, val, str(N), str(K)])
		
	if na_rate != 0:
		prefix += '-na{}'.format(na_rate)
		
	if model_name == 'proto':
		model = Proto(sentence_encoder, hidden_size = hidden_size)
	else:
		raise NotImplementedError
	
	if not os.path.exists('checkpoint'):
		os.mkdir('checkpoint')
	ckpt = 'checkpoint/{}.pth.tar'.format(prefix)
	
	if save_ckpt:
		ckpt = save_ckpt
		
	if torch.cuda.is_available():
		model.cuda()
	
	if not only_test:
		bert_optim = False
		
		framework.train(model, prefix, batch_size, trainN, N, K, Q,
				pytorch_optim = pytorch_optim, load_ckpt = load_ckpt,save_ckpt = save_ckpt,
				na_rate = na_rate, val_step = val_step, fp16 = fp16, pair = pair,
				train_iter = train_iter, val_iter = val_iter, bert_optim = bert_optim)
	else:
		ckpt = load_ckpt
	
	acc = framework.eval(model, batch_size, N, K, Q, test_iter, na_rate = na_rate, ckpt = ckpt, pair = pair)
	print("RESULT: %.2f" % (acc * 100))
	
if __name__ == "__main__":
	main() 
		
				
		
		
		
		
	
