"""
https://huggingface.co/afrideva/TinyLlama-1.1B-intermediate-step-955k-token-2T-guanaco-GGUF

jncraton/TinyLlama-1.1B-intermediate-step-955k-token-2T-guanaco-GGUF
Quantized GGUF model files for TinyLlama-1.1B-intermediate-step-955k-token-2T-guanaco from jncraton
---
I used the Guanaco prompt template format
f"### Human: {prompt} ### Assistant:"

"""
import gradio as gr
from llama_cpp import Llama
import datetime

#MODEL SETTINGS also for DISPLAY
liked = 2
convHistory = ''
convHistory = ''
mrepo = 'afrideva/TinyLlama-1.1B-intermediate-step-955k-token-2T-guanaco-GGUF'
modelfile = "models/tinyllama-1.1b-intermediate-step-955k-token-2t-guanaco.q8_0.gguf"
modeltitle = "TinyLlama-Guanaco"
modelparameters = '1.1B'
model_is_sys = False
modelicon = '🚀'
imagefile = './miniguanaco.png'
repetitionpenalty = 1.2
contextlength=2048
stoptoken = '</s>'
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
f"### Human: {b} ### Assistant:"
"""
def combine(a, b, c, d,e,f):
    global convHistory
    import datetime
    temperature = c
    max_new_tokens = d
    repeat_penalty = f
    top_p = e
    prompt = f"### Human: {b} ### Assistant:"
    start = datetime.datetime.now()
    generation = ""
    delta = ""
    prompt_tokens = f"Prompt Tokens: {len(llm.tokenize(bytes(prompt,encoding='utf-8')))}"
    generated_text = ""
    answer_tokens = ''
    total_tokens = ''   
    for character in llm(prompt, 
                max_tokens=max_new_tokens, 
                stop=['###', stoptoken], #'<|im_end|>'  '#'  '<|endoftext|>'
                temperature = temperature,
                repeat_penalty = repeat_penalty,
                top_p = top_p,   # Example stop token - not necessarily correct for this specific model! Please check before using.
                echo=False, 
                stream=True):
        generation += character["choices"][0]["text"]

        answer_tokens = f"Out Tkns: {len(llm.tokenize(bytes(generation,encoding='utf-8')))}"
        total_tokens = f"Total Tkns: {len(llm.tokenize(bytes(prompt,encoding='utf-8'))) + len(llm.tokenize(bytes(generation,encoding='utf-8')))}"
        delta = datetime.datetime.now() - start
        seconds = delta.total_seconds()
        speed = (len(llm.tokenize(bytes(prompt,encoding='utf-8'))) + len(llm.tokenize(bytes(generation,encoding='utf-8'))))/seconds
        textspeed = f"Gen.Speed: {speed} t/s"
        yield generation, delta, prompt_tokens, answer_tokens, total_tokens, textspeed
    timestamp = datetime.datetime.now()
    textspeed = f"Gen.Speed: {speed} t/s"
    logger = f"""time: {timestamp}\n Temp: {temperature} - MaxNewTokens: {max_new_tokens} - RepPenalty: {repeat_penalty}  Top_P: {top_p}  \nPROMPT: \n{prompt}\n{modeltitle}_{modelparameters}: {generation}\nGenerated in {delta}\nPromptTokens: {prompt_tokens}   Output Tokens: {answer_tokens}  Total Tokens: {total_tokens}  Speed: {speed}\n---"""
    writehistory(logger)
    convHistory = convHistory + prompt + "\n" + generation + "\n"
    print(convHistory)
    return generation, delta, prompt_tokens, answer_tokens, total_tokens, textspeed   
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
            - **Context Lenght**: {contextlength} tokens
            - **LLM Engine**: llama-cpp
            - **Model**: {modelicon} {modeltitle}
            - **Log File**: {logfile}
            """)

            txt_Messagestat = gr.Textbox(value="", placeholder="SYS STATUS:", lines = 1, interactive=False, show_label=False)              
            txt_likedStatus = gr.Textbox(value="", placeholder="Liked status: none", lines = 1, interactive=False, show_label=False)
            txt_speed = gr.Textbox(value="", placeholder="Gen.Speed: none", lines = 1, interactive=False, show_label=False) 


        with gr.Column(scale=4):
            txt = gr.Textbox(label="System Prompt", lines=1, interactive = model_is_sys, value = 'You are an advanced and helpful AI assistant.')
            txt_2 = gr.Textbox(label="User Prompt", lines=5, show_copy_button=True)
            with gr.Row():
                btn = gr.Button(value=f"{modelicon} Generate", variant='primary', scale=3)
                btnlike = gr.Button(value=f"👍 GOOD", variant='secondary', scale=1)
                btndislike = gr.Button(value=f"🤮 BAD", variant='secondary', scale=1)
                submitnotes = gr.Button(value=f"💾 SAVE NOTES", variant='secondary', scale=2)
            txt_3 = gr.Textbox(value="", label="Output", lines = 8, show_copy_button=True)
            txt_notes = gr.Textbox(value="", label="Generation Notes", lines = 2, show_copy_button=True)
                
            def likeGen():
                #set like/dislike and clear the previous Notes
                global liked
                liked = f"👍 GOOD"
                resetnotes = ""
                return liked, resetnotes
            def dislikeGen():
                #set like/dislike and clear the previous Notes
                global liked
                liked = f"🤮 BAD"
                resetnotes = ""
                return liked, resetnotes
            def savenotes(vote,text):
                logging = f"### NOTES AND COMMENTS TO GENERATION\nGeneration Quality: {vote}\nGeneration notes: {text}\n---\n\n"
                writehistory(logging)
                message = "Notes Successfully saved"
                print(logging)
                print(message)
                return message

            btn.click(combine, inputs=[txt, txt_2,temp,max_len,top_p,repPen], outputs=[txt_3,gentime,prompttokens,outputokens,totaltokens,txt_speed])
            btnlike.click(likeGen, inputs=[], outputs=[txt_likedStatus,txt_notes])
            btndislike.click(dislikeGen, inputs=[], outputs=[txt_likedStatus,txt_notes])
            submitnotes.click(savenotes, inputs=[txt_likedStatus,txt_notes], outputs=[txt_Messagestat])



if __name__ == "__main__":
    demo.launch(inbrowser=True)
