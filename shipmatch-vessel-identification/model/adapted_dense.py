"""Generic encoder adapted exclusively on official names, row-local inference."""
from pathlib import Path
import os,json
from dense_names import NameEncoder


class AdaptedNameEncoder(NameEncoder):
    def __init__(self,path,adapter=None,device=None,batch_size=32):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        import torch
        from transformers import AutoTokenizer,AutoModel
        torch.set_num_threads(6);torch.use_deterministic_algorithms(True)
        self.torch=torch;self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size=batch_size;self.cache={}
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True,padding_side='left')
        self.model=AutoModel.from_pretrained(str(path),local_files_only=True,torch_dtype=torch.float16 if self.device=='cuda' else torch.float32,attn_implementation='sdpa').to(self.device).eval()
        if adapter is not None:self.apply_adapter(adapter)

    def apply_adapter(self,path):
        from safetensors.torch import load_file
        path=Path(path);c=json.loads((path/'adapter_config.json').read_text())
        if set(c['target_modules'])!={'q_proj','v_proj'} or c.get('bias')!='none':raise ValueError('Unsupported encoder adapter')
        weights=load_file(str(path/'adapter_model.safetensors'),device=self.device)
        self.model.to(self.torch.bfloat16);base=dict(self.model.named_parameters());scale=c['lora_alpha']/c['r'];count=0
        with self.torch.no_grad():
            for key,a in weights.items():
                if not key.endswith('.lora_A.weight'):continue
                b=weights[key.replace('.lora_A.weight','.lora_B.weight')]
                target=key.removeprefix('base_model.model.').replace('.lora_A.weight','.weight')
                if target not in base:raise ValueError('Encoder adapter target mismatch')
                base[target].add_((b.float()@a.float())*scale);count+=1
        if count*2!=len(weights) or not count:raise ValueError('Incomplete encoder adapter')
        self.model.to(self.torch.float16 if self.device=='cuda' else self.torch.float32).eval();self.cache.clear()
