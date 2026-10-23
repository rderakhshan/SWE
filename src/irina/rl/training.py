"""GRPO-style grouped token objective and trainable LoRA update.

Inputs: generated token records (prompt_ids:list[int], tokens:list[int]),
rollout advantages (list[float]), and QwenProvider. Outputs: a training
metrics dict with loss and updated status. Example:
``advantages([0,1])`` yields approximately ``tensor([-1,1])``.
Policy loss masks prompt/observation tokens by recomputing only generated
continuations. Old log probabilities are evaluated before optimizer update.
"""
import torch

def advantages(rewards,epsilon=1e-8):
    x=torch.tensor(rewards,dtype=torch.float32)
    if len(x)<2 or x.std(unbiased=False).item()<epsilon:return torch.zeros_like(x)
    return (x-x.mean())/(x.std(unbiased=False)+epsilon)

def clipped_loss(current_logps,old_logps,masks,advs,clip=0.2):
    losses=[]
    for now,old,mask,adv in zip(current_logps,old_logps,masks,advs):
        ratio=torch.exp((now-old.detach()).clamp(-20,20))
        objective=torch.minimum(ratio*adv,torch.clamp(ratio,1-clip,1+clip)*adv)
        losses.append(-(objective*mask).sum()/mask.sum().clamp(min=1))
    return torch.stack(losses).mean()

def token_logprobs(model,prompt_ids,generated_ids):
    if not generated_ids:return None
    device=next(model.parameters()).device
    ids=torch.tensor([prompt_ids+generated_ids],device=device)
    outputs=model(input_ids=ids)
    logits=outputs.logits[0,len(prompt_ids)-1:-1,:]
    targets=ids[0,len(prompt_ids):]
    return logits.float().log_softmax(dim=-1).gather(-1,targets.unsqueeze(-1)).squeeze(-1)

def train_group(provider,rollout_records,rewards,learning_rate=1e-4,clip=0.2):
    """Each rollout is a list of generation records; advantages are rollout-level."""
    if len(rollout_records)!=len(rewards):raise ValueError('rollout/reward mismatch')
    adv=advantages(rewards)
    if not torch.any(adv):return {'loss':0.0,'updated':False,'reason':'zero_variance_rewards'}
    model=provider.model
    old=[]
    model.eval()
    with torch.no_grad():
        for group in rollout_records:
            old.append([token_logprobs(model,r['prompt_ids'],r['tokens']) for r in group])
    model.train()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=learning_rate)
    losses=[]
    for group,old_group,a in zip(rollout_records,old,adv):
        for record,old_lp in zip(group,old_group):
            if old_lp is None:continue
            now=token_logprobs(model,record['prompt_ids'],record['tokens'])
            losses.append(clipped_loss([now],[old_lp],[torch.ones_like(now)],[a.to(now.device)],clip))
    if not losses:return {'loss':None,'updated':False,'reason':'no_generated_tokens'}
    loss=torch.stack(losses).mean();loss.backward()
    params=[p for p in model.parameters() if p.requires_grad]
    grad_norm=torch.nn.utils.clip_grad_norm_(params,1.0)
    if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
        opt.zero_grad();raise FloatingPointError('nonfinite loss or gradients')
    opt.step();opt.zero_grad();model.eval()
    return {'loss':float(loss.detach()),'updated':bool(grad_norm>0),'grad_norm':float(grad_norm)}
