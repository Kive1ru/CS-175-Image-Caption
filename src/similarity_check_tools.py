from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils import get_device

class SimilarityTool():
    def __init__(self):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens', device=get_device())
    
    def check_sentences(self, sentences):
        sentence_embeddings = self.model.encode(sentences)
        sentscore = cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])
        sentscore = sentscore.tolist()
        # print(sentscore[0])
        bestind = sentscore[0].index(max(sentscore[0]))
        return (bestind+1,max(sentscore[0]))
        # print("most similar sentence[",sentences[bestind+1], "]  similarity score:[",round(max(sentscore[0])*100,2),"%]")
        # _, ind = sentscore.where(arr == max(sentscore[0]))
        # print(sentscore)
    
    def check_text(self, gcaption, ecaption):
        sentences = [gcaption, ecaption]
        sentence_embeddings = self.model.encode(sentences)
        sentscore = cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])
        sentscore = sentscore.tolist()
        
        return (sentscore[0][0])

# def similarity_check(sentences):
#     wfpsentences = sentences


#     model = SentenceTransformer('bert-base-nli-mean-tokens')
#     sentence_embeddings = model.encode(wfpsentences)

#     sentscore = cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:])
#     print(sentscore)
#     print(cosine_similarity([sentence_embeddings[0]],sentence_embeddings[1:]))
#     # bestind = sentscore.index(max(sentscore))

if __name__ == "__main__":
    texta0="A god going into a wooden cabin"
    texta1="A child in a pink dress is climbing up a set of stairs in an entry way ."
    texta2="A girl going into a wooden building"
    texta3="A little girl climbing into a wooden playhouse"
    texta4="A little girl climbing the stairs to her playhouse"
    texta5="A little girl in a pink dress going into a wooden cabin"
    sentences1 = [texta0,texta1,texta2,texta3,texta4,texta5]

    textb0 = "black dog and dog fighting"
    textb1 = "A black dog and a spotted dog are fighting"
    textb2 = "A black dog and a tri-colored dog playing with each other on the road ."
    textb3 = "A black dog and a white dog with brown spots are staring at each other in the street ."
    textb4 = "Two dogs of different breeds looking at each other on the road ."
    textb5 = "Two dogs on pavement moving toward each other ."
    sentences2 = [textb0,textb1,textb2,textb3,textb4,textb5]

    sentc = SimilarityTool()
    print(sentc.check_sentences(sentences1))
    print(sentc.check_sentences(sentences2))
    print(sentc.check_sentences(["this is me", "this is me", "that is me", "this person is me"]))
    print(sentc.check_sentences(["this is me", "this is not him"]))
    print(sentc.check_text("black dog and dog fighting", "A black dog and a spotted dog are fighting"))