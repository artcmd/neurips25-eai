# neurips25-eai

Embodied Agent Interface (EAI) Challenge @ NeurIPS 2025

## Benchmark

https://embodied-agent-interface.github.io/

benchmark dataset: 338 tasks in VirtualHome, 100 tasks in BEHAVIOR

## Challenge

https://neurips25-eai.github.io/

https://eval.ai/web/challenges/challenge-page/2621/overview


---

- [x] 09/02 started working on this challenge and learning LLM from zero
- [ ] 11/15 last day of the challenge development phase

## set up conda environment
```bash
conda create -n EAIenv python=3.12
conda activate EAIenv
pip install vllm==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.57.1
pip install flashinfer-python==0.4.1
pip install scikit-learn matplotlib pandas
```


---

## Changelog

- 09/23 add Qwen3
- 09/30 add GPT-OSS
- 10/15 update conda environment 
