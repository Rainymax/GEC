import einops
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch
from typing import *

class GECModel(nn.Module):
    """
    使用GRU构建Encoder-Decoder
    Elmo作为Encoder Embedding
    Decoder Embedding 也可以用elmo

    Args:
        elmo_model: ELmo 模型
        vocab_size: 词表大小

    """
    def __init__(self, elmo_model, vocab_size:int):
        super().__init__()
        self.input_size = 1024
        self.hidden_size = 512
        self.vocab_size = vocab_size
        self.elmo = elmo_model
        self.encoder = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size // 2, num_layers=2, batch_first=True, bidirectional=True)
        self.decoder_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.decoder = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=2, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(p=0.1)
        self.transform = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(2 * self.hidden_size, self.vocab_size)

    def encode(self, source_inputs: List[List[str]], source_mask: torch.Tensor, **kwargs):
        """
        Encode input source text
        Args:
            source_inputs: a list of input text
            source_mask: torch.tensor of size (batch_size, sequence_length)
        Returns:
            encoder_outputs: torch.Tensor of size (batch_size, sequence_length, hidden_states)
        """
        raw_embeddings = self.elmo.sents2elmo(source_inputs)
        embeddings = torch.tensor(np.stack(raw_embeddings, axis=0), dtype=torch.float, device=source_mask.device)
        sequence_length = source_mask.sum(dim=1).detach().cpu()
        packed_embeddings = pack_padded_sequence(embeddings, sequence_length, batch_first=True, enforce_sorted=False)
        packed_outputs, _ = self.encoder(packed_embeddings)
        encoder_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        return encoder_outputs.contiguous()

    def decode(self, encoder_outputs: torch.Tensor, source_mask: torch.Tensor, target_input_ids: torch.Tensor, target_inputs: List[List[str]], target_mask: torch.Tensor=None, hidden_states: torch.Tensor=None, **kwargs):
        """
        decode for output sequence
        Args
            encoder_outputs: a torch.Tensor of (batch_size, sequence_length, hidden_states) output by encoder
            source_mask: torch.tensor of size (batch_size, sequence_length)
            target_input_ids: torch.tensor of size (batch_size, sequence_length)
            target_inputs: a list of target text
            target_mask: torch.tensor of size (batch_size, sequence_length)
        Returns:
            decoder_output: a torch.Tensor of (batch_size, sequence_length, vocab_size)
        """
        embeddings = self.dropout(self.decoder_embedding(target_input_ids))
        if target_mask is not None:
            sequence_length = target_mask.sum(dim=1).detach().cpu()
            packed_embeddings = pack_padded_sequence(embeddings, sequence_length, batch_first=True, enforce_sorted=False)
            packed_output, hidden_states = self.decoder(packed_embeddings, hidden_states)
            rnn_outputs, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            rnn_outputs, hidden_states = self.decoder(embeddings, hidden_states)
        # print(encoder_outputs.shape, source_mask.shape, rnn_outputs.shape, hidden_states.shape, target_mask.shape)
        attention_weights = torch.einsum("bsh,beh->bse", rnn_outputs, self.transform(encoder_outputs)) + einops.repeat((1 - source_mask) * -1e9, "b e -> b s e", s=rnn_outputs.shape[1])
        context = torch.einsum("bse,beh->bsh", torch.softmax(attention_weights, dim=-1), encoder_outputs)
        decoder_outputs = self.out(torch.cat([rnn_outputs, context], dim=-1))
        return decoder_outputs

    def forward(self, **kwargs):
        encoder_outputs = self.encode(**kwargs)
        decoder_outputs = self.decode(encoder_outputs, **kwargs)
        return decoder_outputs