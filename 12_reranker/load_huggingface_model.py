from transformers import AutoModel

# model_name = "cross-encoder/ms-marco-TinyBERT-L-2"
model_name = "cross-encoder/ms-marco-MultiBERT-L-12"
# model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# model = AutoModel.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, cache_dir="/tmp")
# model = AutoModel.from_pretrained("ms-marco-MiniLM-L-12-v2", cache_dir="/tmp")
