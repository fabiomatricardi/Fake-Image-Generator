# Fake-Image-Generator
a gradio local app that uses semantic search to return Images from a local database

Code for the series of articles on Medium
- [part 1](https://medium.com/artificial-intel-ligence-playground/fake-image-generator-model-part-1-14f1036d808e?sk=b3d0c95aee1d6be3a47d637f65be70bc)
- [part 2](https://medium.com/artificial-intel-ligence-playground/decd7d2c26f9?sk=f981394b868f967fc85171bec451dc95)

### requirements
```bash
pip install datasets requests tqdm huggingface_hub pandas pyarrow gradio faiss-cpu
```

<img src='https://github.com/fabiomatricardi/Fake-Image-Generator/raw/main/FIG_4_001.gif' width=900>

---
  
### HF repo for the images
Images and prompts from [kaupane/nano-banana-pro-gen](https://huggingface.co/datasets/kaupane/nano-banana-pro-gen)

### How to use it
there are 4 files:
- `1.big_dset_parquet_Ask.py` to download a dataset and the images from Hugging Face [kaupane/nano-banana-pro-gen](https://huggingface.co/datasets/kaupane/nano-banana-pro-gen)
- `2.FAKE_Image_Generator.py` to test de-noise effect with random images from the new deck of pictures you just get.
- `3.create_db.py` to create the vector db with FAISS
- `4.gradio_image_generator.py` the final app, to get inspired and use the images in your essays, blog, email or notes.

> all instructions in the articles (free)

---






---


