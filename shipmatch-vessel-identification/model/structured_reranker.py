"""Fixed-shape offline Qwen structured vessel-pair scorer."""
import os
from neural_reranker import NameReranker, PREFIX, SUFFIX
from shipmatch_v4 import JOINT_INSTRUCTION


class StructuredReranker(NameReranker):
    def __init__(self, path, device=None, batch_size=32, max_length=256):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        torch.set_num_threads(6)
        torch.use_deterministic_algorithms(True)
        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size, self.max_length = batch_size, max_length
        self.cache = {}
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True, torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32,
            attn_implementation='sdpa').to(self.device).eval()
        self.no = self.tokenizer.convert_tokens_to_ids('no')
        self.yes = self.tokenizer.convert_tokens_to_ids('yes')
        self.prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)

    def inputs(self, pairs):
        texts = [f'<Instruct>: {JOINT_INSTRUCTION}\n<Query>: {q}\n<Document>: {c}' for q, c in pairs]
        ids = self.tokenizer(texts, padding=False, truncation=True, add_special_tokens=False,
                             max_length=self.max_length-len(self.prefix)-len(self.suffix))['input_ids']
        ids = [self.prefix + x + self.suffix for x in ids]
        batch = self.tokenizer.pad({'input_ids': ids}, padding='max_length', max_length=self.max_length, return_tensors='pt')
        batch['position_ids'] = (batch['attention_mask'].long().cumsum(-1)-1).clamp_min(0)
        return {k: v.to(self.device) for k, v in batch.items()}
