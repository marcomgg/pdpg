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

Bibtex:
```
@INPROCEEDINGS{9341559,
  author={Maggipinto, Marco and Susto, Gian Antonio and Chaudhari, Pratik},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Proximal Deterministic Policy Gradient}, 
  year={2020},
  volume={},
  number={},
  pages={5438-5444},
  doi={10.1109/IROS45743.2020.9341559}}
  ```
