"""
https://huggingface.co/afrideva/Aira-2-1B1-GGUF
===============================================
afrideva/Aira-2-1B1-GGUF
Aira-2 is the second version of the Aira instruction-tuned series. Aira-2-1B1 is an instruction-tuned GPT-style model based on TinyLlama-1.1B. The model was trained with a dataset composed of prompts and completions generated synthetically by prompting already-tuned models (ChatGPT, Llama, Open-Assistant, etc).

<|startofinstruction|>What is a language model?<|endofinstruction|>A language model is a probability distribution over a vocabulary.<|endofcompletion|>


    bos_token_id=tokenizer.bos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,

f"<|startofinstruction|>{b}<|endofinstruction|>"
mrepo = 'afrideva/Aira-2-1B1-GGUF'
modelfile = "model/aira-2-1b1.q5_k_m.gguf"    

aira_logo1.jpg

    """
import gradio as gr
from llama_cpp import Llama
import datetime

#MODEL SETTINGS also for DISPLAY
liked = 2
convHistory = ''
modelfile = "models/aira-2-1b1.q8_0.gguf"
modeltitle = "aira-2-1b1"
modelparameters = '1B'
model_is_sys = False
modelicon = '🌬️'
imagefile = './aira_logo1.jpg'
repetitionpenalty = 1.2
contextlength=2048
logfile = f'{modeltitle}_logs.txt'
print(f"loading model {modelfile}...")
stt = datetime.datetime.now()
# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = Llama(
  model_path=modelfile,  # Download the model file first
  n_ctx=contextlength,  # The max sequence length to use - note that longer sequence lengths require much more resources
  #n_threads=2,            # The number of CPU threads to use, tailor to your system and the resulting performance
)
dt = datetime.datetime.now() - stt
print(f"Model loaded in {dt}")

def writehistory(text):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(text)
        f.write('\n')
    f.close()

"""
f"<|startofinstruction|>{b}<|endofinstruction|>"
"""
def combine(a, b, c, d,e,f):
    global convHistory
    import datetime
    temperature = c
    max_new_tokens = d
    repeat_penalty = f
    top_p = e
    prompt = f"<|startofinstruction|>{b}<|endofinstruction|>"
    start = datetime.datetime.now()
    generation = ""
    delta = ""
    prompt_tokens = f"Prompt Tokens: {len(llm.tokenize(bytes(prompt,encoding='utf-8')))}"
    generated_text = ""
    answer_tokens = ''
    total_tokens = ''   
    for character in llm(prompt, 
                max_tokens=max_new_tokens, 
                stop=['</s>'], #'<|im_end|>'  '#'  '<|endoftext|>'
                temperature = temperature,
                repeat_penalty = repeat_penalty,
                top_p = top_p,   # Example stop token - not necessarily correct for this specific model! Please check before using.
                echo=False, 
                stream=True):
        generation += character["choices"][0]["text"]

        answer_tokens = f"Out Tkns: {len(llm.tokenize(bytes(generation,encoding='utf-8')))}"
        total_tokens = f"Total Tkns: {len(llm.tokenize(bytes(prompt,encoding='utf-8'))) + len(llm.tokenize(bytes(generation,encoding='utf-8')))}"
        delta = datetime.datetime.now() - start
        yield generation, delta, prompt_tokens, answer_tokens, total_tokens
    timestamp = datetime.datetime.now()
    logger = f"""time: {timestamp}\n Temp: {temperature} - MaxNewTokens: {max_new_tokens} - RepPenalty: {repeat_penalty}  Top_P: {top_p}  \nPROMPT: \n{prompt}\n{modeltitle}_{modelparameters}: {generation}\nGenerated in {delta}\nPromptTokens: {prompt_tokens}   Output Tokens: {answer_tokens}  Total Tokens: {total_tokens}\n---"""
    writehistory(logger)
    convHistory = convHistory + prompt + "\n" + generation + "\n"
    print(convHistory)
    return generation, delta, prompt_tokens, answer_tokens, total_tokens    
    #return generation, delta


# MAIN GRADIO INTERFACE
with gr.Blocks(theme='Medguy/base2') as demo:   #theme=gr.themes.Glass()  #theme='remilia/Ghostly'
    #TITLE SECTION
    with gr.Row(variant='compact'):
            with gr.Column(scale=3):            
                gr.Image(value=imagefile, 
                        show_label = False, height = 160,
                        show_download_button = False, container = False,)              
            with gr.Column(scale=10):
                gr.HTML("<center>"
                + "<h3>Prompt Engineering Playground!</h3>"
                + f"<h1>{modelicon} {modeltitle} - {modelparameters} parameters - {contextlength} context window</h1></center>")  
                with gr.Row():
                        with gr.Column(min_width=80):
                            gentime = gr.Textbox(value="", placeholder="Generation Time:", min_width=50, show_label=False)                          
                        with gr.Column(min_width=80):
                            prompttokens = gr.Textbox(value="", placeholder="Prompt Tkn:", min_width=50, show_label=False)
                        with gr.Column(min_width=80):
                            outputokens = gr.Textbox(value="", placeholder="Output Tkn:", min_width=50, show_label=False)            
                        with gr.Column(min_width=80):
                            totaltokens = gr.Textbox(value="", placeholder="Total Tokens:", min_width=50, show_label=False)   
    # INTERACTIVE INFOGRAPHIC SECTION
    

    # PLAYGROUND INTERFACE SECTION
    with gr.Row():
        with gr.Column(scale=1):
            #gr.Markdown(
            #f"""### Tunning Parameters""")
            temp = gr.Slider(label="Temperature",minimum=0.0, maximum=1.0, step=0.01, value=0.1)
            top_p = gr.Slider(label="Top_P",minimum=0.0, maximum=1.0, step=0.01, value=0.8)
            repPen = gr.Slider(label="Repetition Penalty",minimum=0.0, maximum=4.0, step=0.01, value=1.2)
            max_len = gr.Slider(label="Maximum output lenght", minimum=10,maximum=(contextlength-150),step=2, value=512)          
            gr.Markdown(
            f"""
            - **Prompt Template**: Alpaca instruct
            - **Repetition Penalty**: {repetitionpenalty}
            - **Context Lenght**: {contextlength} tokens
            - **LLM Engine**: llama-cpp
            - **Model**: {modelicon} {modelfile}
            - **Log File**: {logfile}
            """)
            gr.Markdown(
            """Vote, Comment and click the button below""")
            submitnotes = gr.Button(value=f"💾 SAVE NOTES", variant='primary')
            txt_Messagestat = gr.Textbox(value="", placeholder="SYS STATUS:", lines = 1, interactive=False, show_label=False)              
            txt_likedStatus = gr.Textbox(value="", placeholder="Liked status: none", lines = 1, interactive=False, show_label=False) 


        with gr.Column(scale=4):
            txt = gr.Textbox(label="System Prompt", lines=1, interactive = model_is_sys, value = 'You are an advanced and helpful AI assistant.')
            txt_2 = gr.Textbox(label="User Prompt", lines=5, show_copy_button=True)
            with gr.Row():
                btn = gr.Button(value=f"{modelicon} Generate", variant='primary', scale=2)
                btnlike = gr.Button(value=f"👍 GOOD", variant='secondary', scale=1)
                btndislike = gr.Button(value=f"🤮 BAD", variant='secondary', scale=1)
            txt_3 = gr.Textbox(value="", label="Output", lines = 8, show_copy_button=True)
            """
            with gr.Row():
                #with gr.Column():
                btnlike = gr.Button(value=f"👍 GOOD", variant='secondary', scale=1)
                btndislike = gr.Button(value=f"🤮 BAD", variant='secondary', scale=1)
                submitnotes = gr.Button(value=f"💾 SAVE NOTES", variant='primary', scale=2) 
            """
            txt_notes = gr.Textbox(value="", label="Generation Notes", lines = 2, show_copy_button=True)
            """
            txt_likedStatus = gr.Textbox(value="", label="Liked status", lines = 1, interactive=False)
            txt_Messagestat = gr.Textbox(value="", label="SYS STATUS", lines = 1, interactive=False)
            """
                
            def likeGen():
                global liked
                liked = f"👍 GOOD"
                return liked
            def dislikeGen():
                global liked
                liked = f"🤮 BAD"
                return liked
            def savenotes(vote,text):
                logging = f"### NOTES AND COMMENTS TO GENERATION\nGeneration Quality: {vote}\nGeneration notes: {text}\n---\n\n"
                writehistory(logging)
                message = "Notes Successfully saved"
                return message

            btn.click(combine, inputs=[txt, txt_2,temp,max_len,top_p,repPen], outputs=[txt_3,gentime,prompttokens,outputokens,totaltokens])
            btnlike.click(likeGen, inputs=[], outputs=[txt_likedStatus])
            btndislike.click(dislikeGen, inputs=[], outputs=[txt_likedStatus])
            submitnotes.click(savenotes, inputs=[txt_likedStatus,txt_notes], outputs=[txt_Messagestat])



if __name__ == "__main__":
    demo.launch(inbrowser=True)