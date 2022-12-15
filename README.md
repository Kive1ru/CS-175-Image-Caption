# CS-175-Image-Caption
## UCI CS 175 PROJECT IN AI (Final Project)
Generates English caption for a given image.


## Required External Library (Newest Version):
    tensorflow | https://www.tensorflow.org/
    pytorch | https://pytorch.org/ (including torch, torchvision, )
    nltk | https://www.nltk.org/
    pandas | https://pandas.pydata.org/
    numpy | https://numpy.org/
    sklearn | https://scikit-learn.org/stable/index.html
    PIL | https://pillow.readthedocs.io/en/stable/
    matplotlib | https://matplotlib.org/
    SentenceTransformers | https://www.sbert.net/

## Publicly Available Code:
##### 1. Flickr8kDataset https://www.kaggle.com/code/mdteach/torch-data-loader-flicker-8k.
    dataset.py
    Use this public source as an insight and a start point, majority of the code inside dataset.py has been modified. 
    Only kept parts of the structure.
    
##### 2. PyTorch Official Tutorial https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    model.py:
    - PositionalEncoding class
    
    eval.py:
    - save_image_caption function: added 10 lines of code

##### 3. Image Captioning With Attention - Pytorch https://www.kaggle.com/code/mdteach/image-captioning-with-attention-pytorch#6-Visualizing-the-attentions
    model.py: 
    - ResNetEncoder class
    
##### 4. A detailed guide to PyTorch’s nn.Transformer() module. https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    model.py:
    - Img2Cap class
        * __init__ and forward methods: modified 20 lines of code

##### 5. BERT For Measuring Text Similarity https://towardsdatascience.com/bert-for-measuring-text-similarity-eec91c6bf9e1
    similarity_check_tool.py
    - idea inspired by this source. Modified and added approximately 30 lines of code

## Team Written Code:
    model.py: all other classes
    train.py
    dataset.py: all code self-written except borrowing the basic code organization
    eval.py: all other functions
    util.py
    similarity_check_tool.py: all code self-written but inspired by source.
    
## Team Report and Powerpoint:
    Final Report https://docs.google.com/document/d/1OHwihVbzBwEoY7vhDldvtK01cUpWiCgN/edit?usp=sharing&ouid=117238140820586439680&rtpof=true&sd=true
    Final PPT https://docs.google.com/presentation/d/1ow5xYlE_4BGYYjSdj2jCaTdR93sBfy55/edit?usp=sharing&ouid=117238140820586439680&rtpof=true&sd=true
