import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
import random

# 1. Chuẩn bị dữ liệu
class SummarizationDataset(Dataset):
    def __init__(self, articles, summaries, vocab):
        self.articles = articles
        self.summaries = summaries
        self.vocab = vocab

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = torch.tensor([self.vocab[token] for token in self.articles[idx].split()])
        summary = torch.tensor([self.vocab[token] for token in self.summaries[idx].split()])
        return article, summary

def collate_fn(batch):
    articles, summaries = zip(*batch)
    articles_padded = pad_sequence(articles, batch_first=True, padding_value=0)
    summaries_padded = pad_sequence(summaries, batch_first=True, padding_value=0)
    return articles_padded, summaries_padded

# 2. Định nghĩa mô hình
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear((hidden_size * 2) + hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return nn.functional.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + hidden_size * 2, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.attention = Attention(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        
        a = self.attention(hidden, encoder_outputs)
        a = a.unsqueeze(1)
        
        weighted = torch.bmm(a, encoder_outputs)
        lstm_input = torch.cat((embedded, weighted), dim=2)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden.unsqueeze(0), hidden.unsqueeze(0)))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        
        return outputs

# 3. Khởi tạo mô hình và huấn luyện
# Giả sử chúng ta có dữ liệu articles và summaries
articles = ["This is a long article about deep learning.", "Another article about natural language processing."]
summaries = ["Article about deep learning.", "NLP article summary."]

# Xây dựng từ điển
def yield_tokens(data):
    for text in data:
        yield text.split()

vocab = build_vocab_from_iterator(yield_tokens(articles + summaries), specials=["<unk>", "<pad>", "<sos>", "<eos>"])
vocab.set_default_index(vocab["<unk>"])

# Tạo dataset và dataloader
dataset = SummarizationDataset(articles, summaries, vocab)
dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

# Khởi tạo mô hình
VOCAB_SIZE = len(vocab)
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.5

encoder = Encoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
decoder = Decoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(encoder, decoder, device).to(device)

# Định nghĩa loss function và optimizer
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
optimizer = optim.Adam(model.parameters())

# Huấn luyện mô hình
NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        output = model(src, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss/len(dataloader)}")

# 4. Sử dụng mô hình để tóm tắt
def summarize(text):
    model.eval()
    tokens = [vocab[token] for token in text.split()]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    src_len = torch.LongTensor([len(tokens)]).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
    
    trg_indexes = [vocab["<sos>"]]
    for _ in range(50):  # Giới hạn độ dài tóm tắt
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
        pred_token = output.argmax(1).item()
        if pred_token == vocab["<eos>"]:
            break
        trg_indexes.append(pred_token)
    
    trg_tokens = [vocab.get_itos()[i] for i in trg_indexes]
    return ' '.join(trg_tokens[1:])  # Bỏ qua token <sos>

# Ví dụ sử dụng
article = "This is a very long article about deep learning and its applications in natural language processing."
summary = summarize(article)
print(f"Original: {article}")
print(f"Summary: {summary}")