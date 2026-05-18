import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

# Tickers escolhidos: Setor bancário brasileiro (ITUB4, BBDC4, BBAS3)
# Justificativa: Menor volatilidade e forte correlação macroeconômica (taxa de juros/inadimplência)
tickers = ['ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA']

# Coletando de 2015 a 2023. O setor bancário reflete bem os ciclos de crédito neste período.
data = yf.download(tickers, start="2015-01-01", end="2023-12-31")['Close']
data = data.ffill().dropna()

# Normalização [0,1] essencial para a convergência dos gradientes na LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.values)

SEQ_LENGTH = 60  # Janela de arrasto: usando os últimos 60 pregões
NUM_FEATURES = len(tickers) # Agora a porta de entrada tem dimensão 3

def create_sequences(dataset, seq_length):
    X, y = [], []
    for i in range(len(dataset) - seq_length):
        X.append(dataset[i:(i + seq_length), :])
        y.append(dataset[i + seq_length, :]) # Prevendo o dia seguinte para os 3 bancos
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Split 80/20 respeitando a temporalidade
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

class MultivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultivariateLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # batch_first=True alinha os tensores no formato (batch, seq, features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) # Extraindo apenas o último state para previsão
        return out

# Instanciando com input_size=3 e output_size=3
model = MultivariateLSTM(input_size=NUM_FEATURES, hidden_size=64, num_layers=2, output_size=NUM_FEATURES)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    if (epoch+1) % 10 == 0:
        print(f'Época [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.6f}')

# =======================
# Validação e Visualização
# =======================
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)

# Revertendo a escala para visualização em Reais (BRL)
test_predictions_inv = scaler.inverse_transform(test_predictions.numpy())
y_test_inv = scaler.inverse_transform(y_test_tensor.numpy())

# Focando no Itaú (índice 0) para o plot final
Acao_Index = 0 
acao_nome = tickers[Acao_Index].replace('.SA', '')

plt.figure(figsize=(14, 5))
plt.plot(y_test_inv[:, Acao_Index], label=f'Real - {acao_nome}', color='blue')
plt.plot(test_predictions_inv[:, Acao_Index], label=f'Previsão - {acao_nome}', color='red', linestyle='dashed')
plt.title(f'Previsão LSTM Multivariada - {acao_nome} (Baseado no Setor Bancário)')
plt.xlabel('Dias (Base de Teste)')
plt.ylabel('Preço (BRL)')
plt.legend()
plt.grid(True)
plt.show()