import torch
import time
import argparse

from ipex_llm.transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import time
from time import perf_counter

model_id = "NousResearch/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_FP32 = AutoModelForCausalLM.from_pretrained(model_id)
model_INT4 = AutoModelForCausalLM.from_pretrained(model_id,
                                                 trust_remote_code=True,
                                                 load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_id)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")]

from threading import Thread
from transformers import TextIteratorStreamer
from fastapi import FastAPI
import uvicorn
import gradio as gr
app = FastAPI()

def chatbot(model_precision,Question):
    streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens = True)
    
    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    if model_precision == 'FP32':
        model = model_FP32
    else:
        model = model_INT4
    
    
    messages = [{"role":'user', 'content': Question}]
    input_ids = tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt").to(model.device)
    terminators = [tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    st = time.time()

    generate_kwargs = dict(
        {"input_ids": input_ids},
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        streamer=streamer,
        top_p=0.9
    )

    new_token = ""
    t = Thread(target= model.generate, kwargs=generate_kwargs)
    t.start()
    chatbot = []
    response_message = ""
        
    for new_token in streamer:
        #if new_token != '<':
            response_message += new_token
            tokenz = tokenizer.tokenize(response_message)
            num_tokens = len(tokenz)
            et = time.time() - st
            yield response_message, f'token:{num_tokens}   time:{round(et,2)}   token/sec: {round(num_tokens/et,2)}' #round(num_tokens)#/et,2)

demo = gr.Interface(
    max_batch_size=1,
    delete_cache=(5, 5),
    fn=chatbot,
    inputs=[gr.Dropdown(["FP32", "INT4"],value = "INT4", label="Select Precision"),
            gr.Textbox(label="Ask Me Anything",lines=5)],
    outputs=[gr.Textbox(label="Answers"), 
             gr.Textbox(label="Tokens/sec")], 
    allow_flagging=False, 
    title="Intel Pytorch Extension (IPEX-LLM) Chatbot", 
    description="""<center><img src="https://upload.wikimedia.org/wikipedia/commons/6/64/Intel-logo-2022.png" width=200px>
    <h2>Inferenced on <u>Azure Compute D32ds v6</u> with AMX Acceleration</h2></n>
    Using NousResearch/Meta-Llama-3-8B-Instruct</center>""", 
    article="""<h3>Built by Malcolm Chan</h3></n>
    If you have stopped the answer generation abruptly, please press <strong>Clear</strong> to purge the history cache before re-using""",  
    theme=gr.Theme.from_hub('HaleyCH/HaleyCH_Theme')
) 

demo.launch(debug=True, share=True)