# install ipex-llm with 'all' option
!pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu

# transformers>=4.33.0 is required for Llama3 with IPEX-LLM optimizations
!pip install transformers==4.37.0 

!pip install gradio
