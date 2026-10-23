"""Local Qwen2.5-Coder inference and PEFT LoRA policy provider.

Input: Hugging Face model identifier str and Irina messages list[dict].
Output: ``generate`` dict with decoded text and token IDs; generations list
stores prompt_ids and generated tokens for token-level GRPO training.
Example: ``QwenProvider().generate([{'role':'user','content':'hello'}])``
returns model-generated text after downloading weights (GPU recommended).
"""
class QwenProvider:
    def __init__(self,model_id='Qwen/Qwen2.5-Coder-0.5B-Instruct'):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        from peft import LoraConfig,get_peft_model
        self.tokenizer=AutoTokenizer.from_pretrained(model_id)
        base=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype='auto',device_map='auto')
        self.model=get_peft_model(base,LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
        self.model.eval();self.generations=[]
    def generate(self,messages):
        import torch
        prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.tokenizer(prompt,return_tensors='pt').to(next(self.model.parameters()).device)
        with torch.no_grad():
            result=self.model.generate(**inputs,max_new_tokens=192,do_sample=True,
                                       temperature=0.9,top_p=0.95,pad_token_id=self.tokenizer.eos_token_id)
        tokens=result[0,inputs.input_ids.shape[1]:].tolist()
        self.generations.append({'prompt_ids':inputs.input_ids[0].tolist(),'tokens':tokens})
        return {'text':self.tokenizer.decode(tokens,skip_special_tokens=True),'tokens':tokens}
