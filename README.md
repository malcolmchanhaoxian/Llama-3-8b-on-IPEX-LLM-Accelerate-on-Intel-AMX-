![image](https://github.com/user-attachments/assets/3a63c96a-78a0-43d0-807b-8034228439f7)

This is a project to test Intel optimised LLM codebase on Azure General Purpose Compute.
The python codebase can be deployed in any Azure General Purpose Compute Family of Dv6 or Ev6

## Installation
```sh
#Install ipex-llm with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu

#Install Transformers for Llama3 with IPEX-LLM optimizations
pip install transformers==4.37.0 

#Install Gradio for UI Design
pip install gradio
```

## Run Code
### (1) Execute via Jupyter
Ensure you have jupyter installed on your host
Codebase is stored in ipex_llm_chatbot.ipynb

### (2) Execute via Python
```sh
python ipex_llm_chatbot.py
```

## Codebase for Quantization
The project compares the performance of baseline LLM models vs optimised model using [IPEX-LLM](https://github.com/intel-analytics/ipex-llm)
Using a modified "AutomodelforCausalLM" line and enable the conversion to 4-bit

```python
model_id = "NousResearch/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_INT4 = AutoModelForCausalLM.from_pretrained(model_id,
                                                 trust_remote_code=True,
                                                 load_in_4bit=True)
```
## Results
The inference result between default and INT4 performance can be seen below with more than 50% improvement.

### INT4
![image](https://github.com/user-attachments/assets/db453937-7ee1-496a-a250-25a56cfcbcfd)

### Non-optimised
![image](https://github.com/user-attachments/assets/2e4a61c9-027c-4fa4-b151-c50b36149a3d)

## Benchmarks
The below is a benchmark exercise to compare the performance of optimised vs. non-optimised model. The inferencing was also tested on two seperate Azure VM instance SKU (Azure Dav5 and Azure Dv6). Azure Dv6 is powered by Intel 5th Gen Xeon Processor (Emerald Rapids) whereas Azure Dav5 is powered by AMD's 3rd Gen EPYC. _Take note that concurrency is not considered here_

<img src="https://github.com/user-attachments/assets/18d30158-3018-43c7-aed8-0b4bd4726a72" width="750">

- Between optimised (INT4 + AMX) vs. non-optimised (FP32), we observed up to an average of 2.7x better performance on the optimised model.
- Between Dv6 vs. Dav5 (both utilising quantised model - INT4), we observed up to an average of 1.8x better performance on Intel Dv6 vs Dav5.

## Disclaimers / Attribution
This repository is for educational purposes and is a community contribution from repository owner. It is not intended for any commercial purposes.
Credits and attribution should be directed to repository owner and all contributors.
 

