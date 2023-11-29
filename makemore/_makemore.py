import torch
from torch.functional import Tensor
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import Dataset

class CharDataset(Dataset):
	def __init__(self, words:str, chars, max_word_length):
		self.words = words
		self.chars = chars
		self.max_word_length = max_word_length
		self.stoi = {ch:i+1 for i,ch in enumerate(chars)} # {char: integer}
		self.itos = {i:s for s,i in self.stoi.items()} # inverse mapping {integer: char}
	
	def __len__(self):
		return len(self.words)
	
	def contains(self, word):
		return word in self.words
	
	def get_vocab_size(self):
		return len(self.chars) + 1 # all the possible characters and special 0 token

	def get_output_length(self):
		return self.max_word_length + 1 # <START> token followed by words

	def encode(self, word: str) -> Tensor:
		ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
		return ix

	def decode(self, ix: Tensor) -> str:
		return ''.join(self.itos[i] for i in ix)
	
	def __getitem__(self, idx):
		#TODO
		return


class Bigram(nn.Module):
	
	def __init__(self, config):
		super().__init__()
		n = config.vocab_size
		self.logits = nn.Parameter(torch.zeros((n, n)))

	# TODO: I understand nothing here!!!
	def forward(self, idx, targets=None):

         # 'forward pass', lol
		logits = self.logits[idx]

        # if we are given some desired targets also calculate the loss
		loss = None
		if targets is not None:
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

		return logits, loss
