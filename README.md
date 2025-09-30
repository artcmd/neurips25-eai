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

## Qwen3 conda environment
```bash
conda create -n qwen3 python=3.11
conda activate qwen3
pip install vllm==0.10.2 --extra-index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/huggingface/transformers.git@main
pip install scikit-learn matplotlib pandas
```

## GPT-OSS conda environment
```bash
conda create -n gptoss python=3.12
conda activate gptoss
pip install --pre vllm==0.10.1+gptoss --extra-index-url https://wheels.vllm.ai/gpt-oss/ --extra-index-url https://download.pytorch.org/whl/nightly/cu128
pip install scikit-learn matplotlib pandas
```
