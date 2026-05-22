from transformers import BertTokenizerFast, BertModel
import torch

model_path = "/home/soic/Desktop/LightDet/units/model/bert"

tokenizer = BertTokenizerFast.from_pretrained(
    model_path,
    local_files_only=True
)

model = BertModel.from_pretrained(
    model_path,
    local_files_only=True
)

model.eval()

text = "黃色箱子"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

print(outputs.last_hidden_state.shape)