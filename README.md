# positional_embedding  
```
def positional_encoding(seq_len, d_model):
    PE = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.pow(10000.0, -torch.arange(0, d_model, dtype=torch.float) / d_model)
    PE[:, 0::2] = torch.sin(position * div_term[0::2])  # div_term for even indices
    if d_model > 1:  # check if d_model is greater than 1
        PE[:, 1::2] = torch.cos(position * div_term[1::2])  # div_term for odd indices
    return PE
    
    
seq_len = 2000
d_model = 5
pos_encoding = positional_encoding(seq_len = seq_len, d_model = d_model)
plt.figure(figsize=(20,10))

from matplotlib import cm

inferno = cm.get_cmap('inferno')


plt.pcolormesh(pos_encoding, cmap=inferno)
plt.xlabel('Embedding Dimensions')
plt.xlim((0, d_model))
plt.ylim((seq_len, 0))
plt.ylabel('Sequence Length')
plt.colorbar()
plt.show()
```
