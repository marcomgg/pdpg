# Code to reproduce results of Proximal Deterministic Policy Gradient
Paper: [https://arxiv.org/pdf/2008.00759.pdf](https://arxiv.org/pdf/2008.00759.pdf) 

Example usage:

```
python -m policygrad with \
environment="HalfCheetah-v2" \
log_freq=5000 \ 
warmup=10000 \
gpu=0 \
seed=100 \
batch_size=256 \
max_timesteps=2e5
```

