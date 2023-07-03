import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_filters,
                 num_classes, dropout, mode):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.dropout = dropout
        self.mode = mode

        self.embedding = nn.Embedding(
            self.vocab_size, self.embedding_dim, padding_idx=0)
        self.load_embeddings()
        self.conv = nn.ModuleList([
            nn.Conv1d(
                in_channels = self.embedding_dim,
                out_channels = self.num_filters,
                kernel_size=k,
                stride=1)
            for k in self.kernel_sizes])
        self.dropout = nn.Dropout(self.dropout)
        self.fc = nn.Linear(len(self.kernel_sizes) * self.num_filters, self.num_classes)

    def forward(self, x):
        # batch_size, sequence_length = x.shape
        x = self.embedding(x.squeeze(dim=-1)).transpose(1, 2)
        x = [F.relu(conv(x)) for conv in self.conv]
        x = [F.max_pool1d(c, c.size(-1)).squeeze(dim=-1) for c in x]
        x = torch.cat(x, dim=1)
        x = self.fc(self.dropout(x))
        return torch.sigmoid(x).squeeze()

    def load_embeddings(self):
        if 'static' in self.mode:
            # FIXME Load vocab from somewhere
            # self.embedding.weight.data.copy_(vocab.vectors)
            if 'non' not in self.mode:
                self.embedding.weight.data.requires_grad = False
                print('Loaded pretrained embeddings, weights are not trainable.')
            else:
                self.embedding.weight.data.requires_grad = True
                print('Loaded pretrained embeddings, weights are trainable.')
        elif self.mode == 'rand':
            print('Randomly initialized embeddings are used.')
        else:
            raise ValueError('Unexpected value of mode. Please choose from static, nonstatic, rand.')