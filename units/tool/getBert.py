from pathlib import Path
import os
import torch
from transformers import BertTokenizerFast, BertModel

PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
print(f"PROJECT_ROOT: {PROJECT_ROOT}")

model_name = "hfl/chinese-macbert-base"

save_path = PROJECT_ROOT / "LightDet/units/model/bert"
cache_path = PROJECT_ROOT / "LightDet/units/model/bert_cache"

text = "黃色箱子"

save_path.mkdir(parents=True, exist_ok=True)
cache_path.mkdir(parents=True, exist_ok=True)

tokenizer = BertTokenizerFast.from_pretrained(
    model_name,
    cache_dir=str(cache_path)
)

model = BertModel.from_pretrained(
    model_name,
    cache_dir=str(cache_path)
)

tokenizer.save_pretrained(str(save_path))
model.save_pretrained(str(save_path))

print(f"Tokenizer and model saved to: {save_path}")

print("Saved files:")
for p in save_path.iterdir():
    print(p.name)

tokenizer = BertTokenizerFast.from_pretrained(
    str(save_path),
    local_files_only=True
)

model = BertModel.from_pretrained(
    str(save_path),
    local_files_only=True,
    use_safetensors=True
)

model.eval()

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

print(outputs.last_hidden_state.shape)