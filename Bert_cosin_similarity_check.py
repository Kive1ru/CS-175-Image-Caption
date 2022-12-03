import torch#pytorch
from transformers import AutoTokenizer, AutoModel#for embeddings
from sklearn.metrics.pairwise import cosine_similarity#for similarity

#download pretrained model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",)
model = AutoModel.from_pretrained("bert-base-uncased",output_hidden_states=True)

#create embeddings
def get_embeddings(text,token_length):
  tokens=tokenizer(text,max_length=token_length,padding='max_length',truncation=True)
  output=model(torch.tensor(tokens.input_ids).unsqueeze(0),
               attention_mask=torch.tensor(tokens.attention_mask).unsqueeze(0)).hidden_states[-1]
  return torch.mean(output,axis=1).detach().numpy()

#calculate similarity
def calculate_similarity(checktext,text1,text2,text3,text4,text5,token_length=20):
    text0=checktext
    rtl = [text1,text2,text3,text4,text5]
    out1=get_embeddings(text1,token_length=token_length)#create embeddings of text
    out2=get_embeddings(text2,token_length=token_length)#create embeddings of text
    out3=get_embeddings(text3,token_length=token_length)#create embeddings of text
    out4=get_embeddings(text4,token_length=token_length)#create embeddings of text
    out5=get_embeddings(text5,token_length=token_length)#create embeddings of text
    out0=get_embeddings(text0,token_length=token_length)#create embeddings of text
    simscore = []
    sim1= cosine_similarity(out1,out0)[0][0]
    sim2= cosine_similarity(out2,out0)[0][0]
    sim3= cosine_similarity(out3,out0)[0][0]
    sim4= cosine_similarity(out4,out0)[0][0]
    sim5= cosine_similarity(out5,out0)[0][0]
    simscore = [sim1,sim2,sim3,sim4,sim5]
    # print("sentence1:",sim1,"sentence2:",sim2,"sentence3:",sim3,"sentence4:",sim4,"sentence5:",sim5)
    bestind = simscore.index(max(simscore))
    print("Best match sentence:",rtl[bestind],"  similarity score of:", simscore[bestind] )
    # print(simscore.index(max(simscore)))
    # if sim1>sim2:
    #     print('sentence 1 is more similar to input sentence')
    # else:
    #     print('sentence 2 is more similar to input sentence')
    return
    
text1="A child in a pink dress is climbing up a set of stairs in an entry way ."
text2="A girl going into a wooden building"
text3="A little girl climbing into a wooden playhouse"
text4="A little girl climbing the stairs to her playhouse"
text5="A little girl in a pink dress going into a wooden cabin"
text0="a girl walk on the tree"

calculate_similarity(text0,text1,text2,text3,text4,text5)

