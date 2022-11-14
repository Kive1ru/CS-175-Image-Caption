import os
import csv
import tensorflow as tf


def data_load():
    caption_map = {}
    BASE_DIR = f"{os.getcwd()}/data/flickr8k"
    with open(os.path.join(BASE_DIR, 'captions.txt'), 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for line in reader:
            file_name, caption = line[0], line[1]
            img_id = file_name.split('.')[0]
            if img_id not in caption_map:
                caption_map[img_id] = []
            caption_map[img_id].append(caption)
    return caption_map


def data_clean(data_map):
    for key, caption_list in data_map.items():
        for i in range(len(caption_list)):
            caption = caption_list[i]
            caption = caption.split()
            caption = [word.lower() for word in caption]
            caption_list[i] = "startseq " + ' '.join(caption) + " endseq"
    return data_map


def tokenize(data_map):
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    captions = []
    for key, caption_list in data_map.items():
        for caption in caption_list:
            captions.append(caption)
    tokenizer.fit_on_texts(captions)
    vocab_size = len(tokenizer.word_index) + 1
    num_images = len(data_map.keys())
    images =list(data_map.keys())
    train_img = images[:int(num_images * 0.7)]
    test_img = images[int(num_images * 0.7):]
    return tokenizer, vocab_size, train_img, test_img, captions

'''
tokenizer, vocab_size, train_img, test_img, captions = tokenize(data_clean(data_load()))
print(tokenizer.texts_to_sequences([captions[1], captions[2]]))
print(captions[1], captions[2])
'''