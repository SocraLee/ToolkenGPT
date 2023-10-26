
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
extra_tokens = ['askhim']
num_added_tokens = tokenizer.add_tokens(extra_tokens)

inputs = tokenizer('you can askhim', truncation=True, padding=True)
print(inputs)