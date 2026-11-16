# DualRef: Adaptive Hybrid Reasoning and Verification Architecture for Small Language Models

Official code and benchmark artifacts for the paper:
**"DualRef: Adaptive Hybrid Reasoning and Verification Architecture for Small Language Models"**
*Dinh-Hong Vu, Hoai-An Duong, Nhat-Hao Ho — Ton Duc Thang University*

## Repository Structure

```text
dualref/
├── data/
│   ├── custom_40_problems.json   # 40 custom diagnostic math problems (M01-M40)
│   ├── math_1000_ids.txt         # Sampled 1,000 problem IDs from the MATH test set
│   └── gsm8k_1000_ids.txt        # Sampled 1,000 problem IDs from the GSM8K test set
├── evaluation.py                 # Main evaluation pipeline
├── LICENSE                       # MIT License
└── README.md
```

## Reproducibility Notes

- **Generator Model:** `Meta-Llama-3.1-8B-Instruct` via LM Studio OpenAI-compatible local API (`seed = 0`).
- **Neural Critic:** `Llama-3.2-3B-Instruct` fine-tuned on PRM800K (`Q4_K_M` quantization).
- **Sampling Seed:** The random subset sampling seed for the 1,000 MATH and 1,000 GSM8K problems was not separately logged; exact sampled IDs are provided in `data/`.

## Usage

```bash
pip install pandas sympy requests
python evaluation.py
```
