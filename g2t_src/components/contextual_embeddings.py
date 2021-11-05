import torch.nn as nn
import torch
from transformers import AlbertModel, AlbertTokenizer, BartModel, BartTokenizer, BertModel, BertTokenizer, ElectraModel, ElectraTokenizer, GPT2Model, GPT2Tokenizer, RobertaModel, RobertaTokenizer, T5Model, T5Tokenizer,  XLNetTokenizer, XLNetModel
from transformers import BertTokenizerFast, ElectraTokenizerFast
# from kobert_tokenizer import KoBERTTokenizer
import pdb

class AlbertEncoder(nn.Module):
	def __init__(self, Albert_model = 'albert-base-v2', device = 'cuda:0 ', freeze_Albert = False):
		super(AlbertEncoder, self).__init__()
		self.Albert_layer = AlbertModel.from_pretrained(Albert_model)
		self.Albert_tokenizer = AlbertTokenizer.from_pretrained(Albert_model)
		self.device = device
		
		if freeze_Albert:
			for p in self.Albert_layer.parameters():
				p.requires_grad = False
		
	def Albert_input(self, sentences):
		'''
		Preprocess the input sentences using Albert tokenizer and converts them to a torch tensor containing token ids

		'''
		#Tokenize the input sentences for feeding into Albert
		all_tokens  = [['[CLS]'] + self.Albert_tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]

		index_retrieve = []
		for sent in all_tokens:
			cur_ls = []
			for j in range(1, len(sent)):
				if sent[j][0] == '#':
					continue
				else:
					cur_ls.append(j)
			index_retrieve.append(cur_ls)

		#Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		#Convert tokens to token ids
		token_ids = torch.tensor([self.Albert_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		#Obtain attention masks
		pad_token = self.Albert_tokenizer.convert_tokens_to_ids('[PAD]')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths, index_retrieve

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a BERT encoder to obtain contextualized representations of each token
		'''
		#Preprocess sentences
		token_ids, attn_masks, input_lengths, index_retrieve = self.Albert_input(sentences)

		#Feed through bert #수정한 부분
		a = self.Albert_layer(token_ids, attention_mask = attn_masks)
		cont_reps = a[0]

		return cont_reps, input_lengths, token_ids, index_retrieve

class BartEncoder(nn.Module):
	def __init__(self, Bart_model = 'facebook/bart-base', device = 'cuda:0 ', freeze_Bart = False):
		super(BartEncoder, self).__init__()
		self.Bart_layer = BartModel.from_pretrained(Bart_model)
		self.Bart_tokenizer = BartTokenizer.from_pretrained(Bart_model)
		self.device = device
		
		if freeze_Bart:
			for p in self.Bart_layer.parameters():
				p.requires_grad = False
		
	def Bart_input(self, sentences):
		'''
		Preprocess the input sentences using Bart tokenizer and converts them to a torch tensor containing token ids

		'''
		#Tokenize the input sentences for feeding into Bart
		all_tokens  = [['[CLS]'] + self.Bart_tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]

		index_retrieve = []
		for sent in all_tokens:
			cur_ls = []
			for j in range(1, len(sent)):
				if sent[j][0] == '#':
					continue
				else:
					cur_ls.append(j)
			index_retrieve.append(cur_ls)

		#Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		#Convert tokens to token ids
		token_ids = torch.tensor([self.Bart_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		#Obtain attention masks
		pad_token = self.Bart_tokenizer.convert_tokens_to_ids('[PAD]')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths, index_retrieve

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a BERT encoder to obtain contextualized representations of each token
		'''
		#Preprocess sentences
		token_ids, attn_masks, input_lengths, index_retrieve = self.Bart_input(sentences)

		#Feed through bert #수정한 부분
		a = self.Bart_layer(token_ids, attention_mask = attn_masks)
		cont_reps = a[0]

		return cont_reps, input_lengths, token_ids, index_retrieve


class BertEncoder(nn.Module):
	def __init__(self, bert_model = 'bert-base-uncased',device = 'cuda:0 ', freeze_bert = False):
		super(BertEncoder, self).__init__()
		self.bert_layer = BertModel.from_pretrained(bert_model)
		self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
		self.device = device
		
		if freeze_bert:
			for p in self.bert_layer.parameters():
				p.requires_grad = False
		
	def bertify_input(self, sentences):
		'''
		Preprocess the input sentences using bert tokenizer and converts them to a torch tensor containing token ids

		'''
		#Tokenize the input sentences for feeding into BERT
		# pdb.set_trace()
		all_tokens  = [['[CLS]'] + self.bert_tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]

		index_retrieve = []
		for sent in all_tokens:
			cur_ls = []
			for j in range(1, len(sent)):
				if sent[j][0] == '#':
					continue
				else:
					cur_ls.append(j)
			index_retrieve.append(cur_ls)
		
		#Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		#Convert tokens to token ids
		token_ids = torch.tensor([self.bert_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		#Obtain attention masks
		pad_token = self.bert_tokenizer.convert_tokens_to_ids('[PAD]')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths, index_retrieve

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a BERT encoder to obtain contextualized representations of each token
		'''
		#Preprocess sentences
		token_ids, attn_masks, input_lengths, index_retrieve = self.bertify_input(sentences)

		#Feed through bert
		cont_reps, _ = self.bert_layer(token_ids, attention_mask = attn_masks)

		return cont_reps, input_lengths, token_ids, index_retrieve

#small : 256
#base : 768-hidden
#large : 1024
'''
	KoELECTRA-Base
	Electra_model = 'monologg/koelectra-base-discriminator'
	
	KoELECTRA-Base-v2
	Electra_model = 'monologg/koelectra-base-v2-discriminator'

	KoELECTRA-Base-v3
	Electra_model = 'monologg/koelectra-small-v3-discriminator'
'''

class ElectraEncoder(nn.Module):
	def __init__(self, Electra_model = 'google/electra-base-discriminator', device = 'cuda:0 ', freeze_Electra = False):
		super(ElectraEncoder, self).__init__()
		self.Electra_layer = ElectraModel.from_pretrained(Electra_model)
		self.Electra_tokenizer = ElectraTokenizer.from_pretrained(Electra_model)
		self.device = device
		
		if freeze_Electra:
			for p in self.Electra_layer.parameters():
				p.requires_grad = False
		
	def Electra_input(self, sentences):
		'''
		Preprocess the input sentences using Electra tokenizer and converts them to a torch tensor containing token ids

		'''
		#Tokenize the input sentences for feeding into Electra
		all_tokens  = [['[CLS]'] + self.Electra_tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]

		index_retrieve = []
		for sent in all_tokens:
			cur_ls = []
			for j in range(1, len(sent)):
				if sent[j][0] == '#':
					continue
				else:
					cur_ls.append(j)
			index_retrieve.append(cur_ls)

		#Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		#Convert tokens to token ids
		token_ids = torch.tensor([self.Electra_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		#Obtain attention masks
		pad_token = self.Electra_tokenizer.convert_tokens_to_ids('[PAD]')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths, index_retrieve

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a BERT encoder to obtain contextualized representations of each token
		'''
		#Preprocess sentences
		token_ids, attn_masks, input_lengths, index_retrieve = self.Electra_input(sentences)

		#Feed through bert #수정한 부분
		a = self.Electra_layer(token_ids, attention_mask = attn_masks)
		cont_reps = a[0]

		return cont_reps, input_lengths, token_ids, index_retrieve


class GPT2Encoder(nn.Module):
	def __init__(self, GPT2_model = 'gpt2', device = 'cuda:0 ', freeze_GPT2 = False):
		super(GPT2Encoder, self).__init__()
		self.GPT2_layer = GPT2Model.from_pretrained(GPT2_model)
		self.GPT2_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_model)
		self.device = device
		
		if freeze_GPT2:
			for p in self.GPT2_layer.parameters():
				p.requires_grad = False
		
	def GPT2_input(self, sentences):
		'''
		Preprocess the input sentences using GPT2 tokenizer and converts them to a torch tensor containing token ids

		'''
		#Tokenize the input sentences for feeding into GPT2
		all_tokens  = [['[CLS]'] + self.GPT2_tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]

		index_retrieve = []
		for sent in all_tokens:
			cur_ls = []
			for j in range(1, len(sent)):
				if sent[j][0] == '#':
					continue
				else:
					cur_ls.append(j)
			index_retrieve.append(cur_ls)

		#Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		#Convert tokens to token ids
		token_ids = torch.tensor([self.GPT2_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		#Obtain attention masks
		pad_token = self.GPT2_tokenizer.convert_tokens_to_ids('[PAD]')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths, index_retrieve

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a BERT encoder to obtain contextualized representations of each token
		'''
		#Preprocess sentences
		token_ids, attn_masks, input_lengths, index_retrieve = self.GPT2_input(sentences)

		#Feed through bert #수정한 부분
		a = self.GPT2_layer(token_ids, attention_mask = attn_masks)
		cont_reps = a[0]

		return cont_reps, input_lengths, token_ids, index_retrieve


class RobertaEncoder(nn.Module):
	def __init__(self, roberta_model = 'roberta-base', device = 'cuda:0 ', freeze_roberta = False):
		super(RobertaEncoder, self).__init__()
		self.roberta_layer = RobertaModel.from_pretrained(roberta_model)
		self.roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model)
		self.device = device
		
		if freeze_roberta:
			for p in self.roberta_layer.parameters():
				p.requires_grad = False
		
	def robertify_input(self, sentences):
		'''
		Preprocess the input sentences using roberta tokenizer and converts them to a torch tensor containing token ids

		'''
		# Tokenize the input sentences for feeding into RoBERTa
		all_tokens  = [['<s>'] + self.roberta_tokenizer.tokenize(sentence) + ['</s>'] for sentence in sentences]
		
		index_retrieve = []
		for sent in all_tokens:
			cur_ls = [1]
			for j in range(2, len(sent)):
				if sent[j][0] == '\u0120':
					cur_ls.append(j)
			index_retrieve.append(cur_ls)				
		
		# Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['<pad>' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		# Convert tokens to token ids
		token_ids = torch.tensor([self.roberta_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		# Obtain attention masks
		pad_token = self.roberta_tokenizer.convert_tokens_to_ids('<pad>')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths, index_retrieve

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a RoBERTa encoder to obtain contextualized representations of each token
		'''
		# Preprocess sentences
		token_ids, attn_masks, input_lengths, index_retrieve = self.robertify_input(sentences)

		# Feed through RoBERTa
		cont_reps, _ = self.roberta_layer(token_ids, attention_mask = attn_masks)

		return cont_reps, input_lengths, token_ids, index_retrieve

#small : 512
#base : 768-hidden
class T5Encoder(nn.Module):
	def __init__(self, T5_model = 't5-base', device = 'cuda:0 ', freeze_T5 = False):
		super(T5Encoder, self).__init__()
		self.T5_layer = T5Model.from_pretrained(T5_model)
		self.T5_tokenizer = T5Tokenizer.from_pretrained(T5_model)
		self.device = device
		
		if freeze_T5:
			for p in self.T5_layer.parameters():
				p.requires_grad = False
		
	def T5_input(self, sentences):
		'''
		Preprocess the input sentences using T5 tokenizer and converts them to a torch tensor containing token ids

		'''
		#Tokenize the input sentences for feeding into T5
		all_tokens  = [['[CLS]'] + self.T5_tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]

		index_retrieve = []
		for sent in all_tokens:
			cur_ls = []
			for j in range(1, len(sent)):
				if sent[j][0] == '#':
					continue
				else:
					cur_ls.append(j)
			index_retrieve.append(cur_ls)

		#Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		#Convert tokens to token ids
		token_ids = torch.tensor([self.T5_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		#Obtain attention masks
		pad_token = self.T5_tokenizer.convert_tokens_to_ids('[PAD]')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths, index_retrieve

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a BERT encoder to obtain contextualized representations of each token
		'''
		#Preprocess sentences
		token_ids, attn_masks, input_lengths, index_retrieve = self.T5_input(sentences)

		#Feed through bert #수정한 부분
		a = self.T5_layer(token_ids, attention_mask = attn_masks)
		cont_reps = a[0]

		return cont_reps, input_lengths, token_ids, index_retrieve

#base : 768-hidden 
#large : 1024
class XLNetEncoder(nn.Module):
	def __init__(self, XLNet_model = 'xlnet-base-cased', device = 'cuda:0 ', freeze_XLNet = False):
		super(XLNetEncoder, self).__init__()
		self.XLNet_layer = XLNetModel.from_pretrained(XLNet_model)
		self.XLNet_tokenizer = XLNetTokenizer.from_pretrained(XLNet_model)
		self.device = device
		
		if freeze_XLNet:
			for p in self.XLNet_layer.parameters():
				p.requires_grad = False
		
	def XLNet_input(self, sentences):
		'''
		Preprocess the input sentences using XLNet tokenizer and converts them to a torch tensor containing token ids

		'''
		#Tokenize the input sentences for feeding into XLNet
		all_tokens  = [['[CLS]'] + self.XLNet_tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]

		index_retrieve = []
		for sent in all_tokens:
			cur_ls = []
			for j in range(1, len(sent)):
				if sent[j][0] == '#':
					continue
				else:
					cur_ls.append(j)
			index_retrieve.append(cur_ls)

		#Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		#Convert tokens to token ids
		token_ids = torch.tensor([self.XLNet_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		#Obtain attention masks
		pad_token = self.XLNet_tokenizer.convert_tokens_to_ids('[PAD]')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths, index_retrieve

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a XLNet encoder to obtain contextualized representations of each token
		'''
		#Preprocess sentences
		token_ids, attn_masks, input_lengths, index_retrieve = self.XLNet_input(sentences)

		#Feed through bert #수정한 부분
		a = self.XLNet_layer(token_ids, attention_mask = attn_masks)
		cont_reps = a[0]

		return cont_reps, input_lengths, token_ids, index_retrieve

class KoBERTEncoder(nn.Module):
	def __init__(self, KoBERT_model = 'skt/kobert-base-v1', device = 'cuda:0 ', freeze_KoBERT = False):
		super(KoBERTEncoder, self).__init__()
		self.KoBERT_layer = BertModel.from_pretrained(KoBERT_model)
		self.KoBERT_tokenizer = KoBERTTokenizer.from_pretrained(KoBERT_model)
		self.device = device
		
		if freeze_KoBERT:
			for p in self.KoBERT_layer.parameters():
				p.requires_grad = False
		
	def KoBERT_input(self, sentences):
		'''
		Preprocess the input sentences using KoBERT tokenizer and converts them to a torch tensor containing token ids

		'''
		#Tokenize the input sentences for feeding into KoBERT
		all_tokens  = [['[CLS]'] + self.KoBERT_tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]

		index_retrieve = []
		for sent in all_tokens:
			cur_ls = []
			for j in range(1, len(sent)):
				if sent[j][0] == '#':
					continue
				else:
					cur_ls.append(j)
			index_retrieve.append(cur_ls)

		#Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		#Convert tokens to token ids
		token_ids = torch.tensor([self.KoBERT_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		#Obtain attention masks
		pad_token = self.KoBERT_tokenizer.convert_tokens_to_ids('[PAD]')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths, index_retrieve

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a KoBERT encoder to obtain contextualized representations of each token
		'''
		#Preprocess sentences
		token_ids, attn_masks, input_lengths, index_retrieve = self.KoBERT_input(sentences)

		#Feed through bert #수정한 부분
		a = self.KoBERT_layer(token_ids, attention_mask = attn_masks)
		cont_reps = a[0]

		return cont_reps, input_lengths, token_ids, index_retrieve




#########
# LMkor #
#########

# albert-base-kor
class LMkorAlbertEncoder(nn.Module):
	def __init__(self, LMkorAlbert_model = "kykim/albert-kor-base", device = 'cuda:0 ', freeze_LMkorAlbert = False):
		super(LMkorAlbertEncoder, self).__init__()
		self.LMkorAlbert_layer = AlbertModel.from_pretrained(LMkorAlbert_model)
		self.LMkorAlbert_tokenizer = BertTokenizerFast.from_pretrained(LMkorAlbert_model)
		self.device = device
		
		if freeze_LMkorAlbert:
			for p in self.LMkorAlbert_layer.parameters():
				p.requires_grad = False
		
	def LMkorAlbert_input(self, sentences):
		'''
		Preprocess the input sentences using LMkorAlbert tokenizer and converts them to a torch tensor containing token ids

		'''
		#Tokenize the input sentences for feeding into LMkorAlbert
		all_tokens  = [['[CLS]'] + self.LMkorAlbert_tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]

		index_retrieve = []
		for sent in all_tokens:
			cur_ls = []
			for j in range(1, len(sent)):
				if sent[j][0] == '#':
					continue
				else:
					cur_ls.append(j)
			index_retrieve.append(cur_ls)

		#Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		#Convert tokens to token ids
		token_ids = torch.tensor([self.LMkorAlbert_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		#Obtain attention masks
		pad_token = self.LMkorAlbert_tokenizer.convert_tokens_to_ids('[PAD]')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths, index_retrieve

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a LMkorAlbert encoder to obtain contextualized representations of each token
		'''
		#Preprocess sentences
		token_ids, attn_masks, input_lengths, index_retrieve = self.LMkorAlbert_input(sentences)

		#Feed through bert #수정한 부분
		a = self.LMkorAlbert_layer(token_ids, attention_mask = attn_masks)
		cont_reps = a[0]

		return cont_reps, input_lengths, token_ids, index_retrieve


# bert-base-kor
class LMkorBERTEncoder(nn.Module):
	def __init__(self, LMkorBERT_model = "kykim/bert-kor-base", device = 'cuda:0 ', freeze_LMkorBERT = False):
		super(LMkorBERTEncoder, self).__init__()
		self.LMkorBERT_layer = BertModel.from_pretrained(LMkorBERT_model)
		self.LMkorBERT_tokenizer = BertTokenizerFast.from_pretrained(LMkorBERT_model)
		self.device = device
		
		if freeze_LMkorBERT:
			for p in self.LMkorBERT_layer.parameters():
				p.requires_grad = False
		
	def LMkorBERT_input(self, sentences):
		'''
		Preprocess the input sentences using LMkorBERT tokenizer and converts them to a torch tensor containing token ids

		'''
		#Tokenize the input sentences for feeding into LMkorBERT
		all_tokens  = [['[CLS]'] + self.LMkorBERT_tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]

		index_retrieve = []
		for sent in all_tokens:
			cur_ls = []
			for j in range(1, len(sent)):
				if sent[j][0] == '#':
					continue
				else:
					cur_ls.append(j)
			index_retrieve.append(cur_ls)

		#Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		#Convert tokens to token ids
		token_ids = torch.tensor([self.LMkorBERT_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		#Obtain attention masks
		pad_token = self.LMkorBERT_tokenizer.convert_tokens_to_ids('[PAD]')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths, index_retrieve

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a LMkorBERT encoder to obtain contextualized representations of each token
		'''
		#Preprocess sentences
		token_ids, attn_masks, input_lengths, index_retrieve = self.LMkorBERT_input(sentences)

		#Feed through bert #수정한 부분
		a = self.LMkorBERT_layer(token_ids, attention_mask = attn_masks)
		cont_reps = a[0]

		return cont_reps, input_lengths, token_ids, index_retrieve


# electra-base-kor
class LMkorElectraEncoder(nn.Module):
	def __init__(self, LMkorElectra_model = "kykim/electra-kor-base", device = 'cuda:0 ', freeze_LMkorElectra = False):
		super(LMkorElectraEncoder, self).__init__()
		self.LMkorElectra_layer = ElectraModel.from_pretrained(LMkorElectra_model)
		self.LMkorElectra_tokenizer = ElectraTokenizerFast.from_pretrained(LMkorElectra_model)
		self.device = device
		
		if freeze_LMkorElectra:
			for p in self.LMkorElectra_layer.parameters():
				p.requires_grad = False
		
	def LMkorElectra_input(self, sentences):
		'''
		Preprocess the input sentences using LMkorElectra tokenizer and converts them to a torch tensor containing token ids

		'''
		#Tokenize the input sentences for feeding into LMkorElectra
		all_tokens  = [['[CLS]'] + self.LMkorElectra_tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]

		index_retrieve = []
		for sent in all_tokens:
			cur_ls = []
			for j in range(1, len(sent)):
				if sent[j][0] == '#':
					continue
				else:
					cur_ls.append(j)
			index_retrieve.append(cur_ls)

		#Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		#Convert tokens to token ids
		token_ids = torch.tensor([self.LMkorElectra_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		#Obtain attention masks
		pad_token = self.LMkorElectra_tokenizer.convert_tokens_to_ids('[PAD]')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths, index_retrieve

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a LMkorElectra encoder to obtain contextualized representations of each token
		'''
		#Preprocess sentences
		token_ids, attn_masks, input_lengths, index_retrieve = self.LMkorElectra_input(sentences)

		#Feed through bert #수정한 부분
		a = self.LMkorElectra_layer(token_ids, attention_mask = attn_masks)
		cont_reps = a[0]

		return cont_reps, input_lengths, token_ids, index_retrieve


class LMkorGPT3Encoder(nn.Module):
	def __init__(self, LMkorGPT3_model = "kykim/gpt3-kor-small_based_on_gpt2", device = 'cuda:0 ', freeze_LMkorGPT3 = False):
		super(LMkorGPT3Encoder, self).__init__()
		self.LMkorGPT3_layer = GPT2LMHeadModel.from_pretrained(LMkorGPT3_model)
		self.LMkorGPT3_tokenizer = BertTokenizerFast.from_pretrained(LMkorGPT3_model)
		self.device = device
		
		if freeze_LMkorGPT3:
			for p in self.LMkorGPT3_layer.parameters():
				p.requires_grad = False
		
	def LMkorGPT3_input(self, sentences):
		'''
		Preprocess the input sentences using LMkorGPT3 tokenizer and converts them to a torch tensor containing token ids

		'''
		#Tokenize the input sentences for feeding into LMkorGPT3
		all_tokens  = [['[CLS]'] + self.LMkorGPT3_tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]

		index_retrieve = []
		for sent in all_tokens:
			cur_ls = []
			for j in range(1, len(sent)):
				if sent[j][0] == '#':
					continue
				else:
					cur_ls.append(j)
			index_retrieve.append(cur_ls)

		#Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		#Convert tokens to token ids
		token_ids = torch.tensor([self.LMkorGPT3_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		#Obtain attention masks
		pad_token = self.LMkorGPT3_tokenizer.convert_tokens_to_ids('[PAD]')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths, index_retrieve

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a LMkorGPT3 encoder to obtain contextualized representations of each token
		'''
		#Preprocess sentences
		token_ids, attn_masks, input_lengths, index_retrieve = self.LMkorGPT3_input(sentences)

		#Feed through bert #수정한 부분
		a = self.LMkorGPT3_layer(token_ids, attention_mask = attn_masks)
		cont_reps = a[0]

		return cont_reps, input_lengths, token_ids, index_retrieve

