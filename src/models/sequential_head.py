import copy
import torch
import torch.nn as nn
from models._abstract_model import SequentialRecModel
from models._modules import LayerNorm, SequentialBlock
from sympy import sequence

print(torch.cuda.is_available())
print(torch.version.cuda)

class SequentialEncoder(nn.Module):
    def __init__(self, args):
        super(SequentialEncoder, self).__init__()
        self.args = args
        self.blocks = nn.ModuleList([SequentialBlock(args) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [ hidden_states ]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])
        return all_encoder_layers

class SequentialModel(SequentialRecModel):
    def __init__(self, args, item_embeddings, user_embeddings=None, pre_trained_embeddings=None):
        super(SequentialModel, self).__init__(args, pre_trained_embeddings=pre_trained_embeddings)
        self.args = args
        self.item_embeddings = item_embeddings.to(self.args.device)
        if user_embeddings is not None:
            self.user_embeddings = user_embeddings.to(self.args.device)
        else:
            self.user_embeddings = user_embeddings
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = SequentialEncoder(args)
        self.apply(self.init_weights)

    def forward(self, input_ids, user_ids=None, all_sequence_output=False):
        if self.user_embeddings is not None:
            # Expand user_ids to match batch size and add an extra dimension
            #user_ids = user_ids.unsqueeze(1)  # Shape: (batch_size, 1)
            # Concatenate user_ids with input_ids
            # Replace the first element of input_ids with user_ids
            input_ids[:, 0] = user_ids  # Replace the first column of input_ids in each batch
            sequence = input_ids
            # sequence = torch.cat([user_ids, input_ids], dim=1)  # Add at the beginning
            # sequence = torch.cat([input_ids, user_ids], dim=1)  # Add at the end
        else:
            sequence = input_ids

        extended_attention_mask = self.get_attention_mask(sequence)
        sequence_emb = self.add_position_embedding(sequence)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        if all_sequence_output:
            sequence_output = item_encoded_layers
        else:
            sequence_output = item_encoded_layers[-1]

        return sequence_output

    def calculate_loss(self, input_ids, answers, neg_answers, same_target, user_ids):
        input_ids = input_ids.to(self.args.device)
        answers = answers.to(self.args.device)
        neg_answers = neg_answers.to(self.args.device)
        user_ids = user_ids.to(self.args.device)
        seq_output = self.forward(input_ids, user_ids=user_ids)
        seq_output = seq_output[:, -1, :]
        item_emb = self.item_embeddings.weight
        logits = torch.matmul(seq_output, item_emb.transpose(0, 1))
        loss = nn.CrossEntropyLoss()(logits, answers)

        return loss

