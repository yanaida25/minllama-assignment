
import torch
import torch.nn.functional as F

# change it with respect to the original model
from config import LlamaConfig
from llama import load_pretrained
from tokenizer import Tokenizer

class LlamaZeroShotClassifier(torch.nn.Module):
	def __init__(self, config: LlamaConfig, tokenizer: Tokenizer, label_names: list[str]):
		super(LlamaZeroShotClassifier, self).__init__()
		self.num_labels = config.num_labels
		self.llama = load_pretrained(config.pretrained_model_path)
		# Zero-shot classification does not require updating llama paramters.
		for param in self.llama.parameters():
			param.requires_grad = False
		assert len(label_names) == self.num_labels
		self.tokenizer = tokenizer
		self.label_name_ids = [tokenizer.encode(label, bos=False, eos=False) for label in label_names]


	def forward(self, input_ids):
		# compute the completion probability of each label string
		logits, _ = self.llama(input_ids)
		log_probabilities = F.log_softmax(logits, dim=-1)
		label_probabilities = torch.zeros((log_probabilities.shape[0], self.num_labels), device=log_probabilities.device)
		for i, label_token_ids in enumerate(self.label_name_ids):
			total_log_prob = torch.sum(log_probabilities[:, :, label_token_ids], axis=-1)
			label_probabilities[:, i] = total_log_prob[:, 0]
		return label_probabilities

class LlamaEmbeddingClassifier(torch.nn.Module):
	def __init__(self, config):
		super(LlamaEmbeddingClassifier, self).__init__()
		self.num_labels = config.num_labels
		self.llama = load_pretrained(config.pretrained_model_path)
		# If we use pretrain mode, we freeze Llama parameters.
		for param in self.llama.parameters():
			if config.option == 'pretrain':
				param.requires_grad = False
			elif config.option == 'finetune':
				param.requires_grad = True

		self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
		self.classifier_head = torch.nn.Linear(self.llama.config.dim, self.num_labels)


	def forward(self, input_ids):
		'''
		1) Find the hidden state after the final token of the input sequence
		2) Apply dropout (self.dropout) to the hidden state at training time to mitigate
		   overfitting.
		2) Pass this through the classifier head (self.classifier_head), which will return
		   logits (unnormalized probabilities) over all classes.
		3) Take the log-softmax of the logits and return log-probabilities over all classes.
		'''

		if not isinstance(input_ids, torch.Tensor):
			input_ids = torch.as_tensor(input_ids) #BS型であるべき
		# モデル（Llama）と同じデバイス（MPS/CPU）に送る
		device = next(self.parameters()).device
		input_ids = input_ids.to(device)
		if input_ids.dim() == 1:
			input_ids = input_ids.unsqueeze(0)
		# print(f"Input IDs : {input_ids.shape}")

		# hidden_states, _ = self.llama(input_ids) # hidden_stateはBSD想定

		logits_voc, hidden_states = self.llama(input_ids)
		# print('outputはOK！！！！！！')
		# print('shape of 秘伝 states from Llama:', hidden_states.shape)
		# pad_id = self.llama.config.pad_token_id
		pad_id = 3  # stories42M.ptのときのpad_id, ハードコーディングだけどさ

		assert pad_id is not None, "pad_id is None: attention_mask cannot be constructed"
		# attention_mask = (input_ids != pad_id).int()
		condition = (input_ids != pad_id)
		# attention_mask = torch.tensor(condition, dtype=torch.long)
		attention_mask = condition.clone().detach()
		sequence_lengths = attention_mask.sum(dim=1) - 1

		batch_size = input_ids.shape[0]
		batch_indices = torch.arange(batch_size, device=input_ids.device)
		# final_hidden_state = hidden_states[batch_indices, sequence_lengths, :].to(torch.float32)
		# print(f"Final hidden state: {final_hidden_state}")
		# print(f"Final hidden state shape: {final_hidden_state.shape}")
		final_hidden_state = hidden_states[:, 1, :] #どうやって前処理しました？？
	
		# 4) 分類処理
		dropped_hidden_state = self.dropout(final_hidden_state)
		logits = self.classifier_head(dropped_hidden_state)
		log_probabilities = F.log_softmax(logits, dim=-1)
		
		return log_probabilities
		# raise NotImplementedError