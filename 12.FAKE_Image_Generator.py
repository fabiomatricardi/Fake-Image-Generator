# starting point is:
# dataset from kaupane/nano-banana-pro-gen
# json prepared from  1.big_dset_parquet_Ask.py  and saved in kaupane_nano-banana-pro-gen.json
# https://huggingface.co/spaces/gradio/fake_diffusion/blob/main/app.py
import gradio as gr
import numpy as np
import time
from PIL import Image
import random



# define core fn, which returns a generator {steps} times before returning the image
def fake_diffusion(steps):
    for _ in range(steps):
        time.sleep(0.21)
        image = np.random.random((486, 896, 4))
        yield image

    # Open the image file
    random_int = random.randint(2800, 3699) # Random int between 2800 and 3699
    img = Image.open(f'images/{random_int}.png') # Replace with your image path

    # Convert the PIL image object to a NumPy array
    image = np.asarray(img) #
    yield image

with gr.Blocks() as demo:
    gr.Markdown("# 🖼️ Fake Image Generation")
    gr.Markdown("")
    gr.Markdown("#### 👾 with diffusion noise effect")
    with gr.Row():
        with gr.Column(scale=1):
            myslide = gr.Slider(1, 50, 10, precision=0)
            submitta = gr.Button("Generate")
        with gr.Column(scale=2):
            myimage = gr.Image(None)

    submitta.click(
        fn=fake_diffusion,
        inputs=myslide,
        outputs=myimage)

#======================
#LAUNCH
#======================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 FAKE IMAGE GENERATION SYSTEM - READY")
    demo.queue()
    demo.launch(server_port=7960,inbrowser=True)
