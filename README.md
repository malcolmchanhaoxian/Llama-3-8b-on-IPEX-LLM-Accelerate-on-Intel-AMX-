![image](https://github.com/user-attachments/assets/3a63c96a-78a0-43d0-807b-8034228439f7)

This is a project to test Intel optimised LLM codebase on Azure General Purpose Compute.
The python codebase can be deployed in any Azure General Purpose Compute Family of Dv6 or Ev6

## Installation

Install all the necessary packages via requirements.py file
```sh
python requirements.py
```
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


