import keras_nlp
import tensorflow as tf
from tensorflow import keras
import random

# prep data by shuffling, splitting into train and val files
# job_data = open('neat_jobs.txt', 'r').read().splitlines()
#
# random.seed(11)
# random.shuffle(job_data)
# train_split = int(0.8*len(job_data))
# test_split = int(0.9*len(job_data))
# ex_train = job_data[:train_split]
# ex_val = job_data[train_split:]
# # ex_test = test_data[test_split:]
# with open('job_train.txt', 'w') as f:
#     f.writelines(["".join([x, '\n'])for x in ex_train])
# with open('job_val.txt', 'w') as f:
#     f.writelines(["".join([x, '\n']) for x in ex_val])

# keras starts here
# data params
BATCH_SIZE = 32
SEQ_LEN = 128
MIN_TRAINING_SEQ_LEN = 300

# model params
EMBED_DIM = 256
FEED_FORWARD_DIM = 256
NUM_HEADS = 3
NUM_LAYERS = 2
VOCAB_SIZE = 5000

# training
EPOCHS = 20

# generating text
GEN_TOKENS = 100

# load and split data
pre_train_ds = (
    tf.data.TextLineDataset('job_train.txt')
    .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=256)
)

pre_val_ds = (
    tf.data.TextLineDataset('job_val.txt')
    .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
    .batch(BATCH_SIZE)
    .shuffle(buffer_size=256)
)

# train tokenizer
vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
    pre_train_ds,
    vocabulary_size=VOCAB_SIZE,
    reserved_tokens=["[PAD]", "[UNK]", "[BOS]"],
)

# load tokenizer
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    sequence_length=SEQ_LEN,
)


start_packer = keras_nlp.layers.StartEndPacker(
    sequence_length=SEQ_LEN,
    start_value=tokenizer.token_to_id("[BOS]"),
)


def preprocess(inputs):
    outputs = tokenizer(inputs)
    features = start_packer(outputs)
    labels = outputs
    return features, labels


# split into train and label sequences
train_ds = pre_train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds = pre_val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

# build the model!
inputs = keras.layers.Input(shape=(None,), dtype=tf.int32)
embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=SEQ_LEN,
    embedding_dim=EMBED_DIM,
    mask_zero=True
)
x = embedding_layer(inputs)
for _ in range(NUM_LAYERS):
    decoder_layer = keras_nlp.layers.TransformerDecoder(
        num_heads=NUM_HEADS,
        intermediate_dim=FEED_FORWARD_DIM,
    )
    x = decoder_layer(x)

outputs = keras.layers.Dense(VOCAB_SIZE)(x)
model = keras.Model(inputs=inputs, outputs=outputs)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)
model.compile(optimizer="adam", loss=loss_fn, metrics=[perplexity])

model.summary()

model.fit(train_ds, validation_data=val_ds, verbose=2, epochs=EPOCHS)

prompt_tokens = start_packer(tokenizer([""]))
prompt_tokens


def next(prompt, cache, index):
    logits = model(prompt)[:, index - 1, :]
    hidden_states = None
    return logits, hidden_states, cache


sampler = keras_nlp.samplers.GreedySampler()
output_tokens = sampler(next=next, prompt=prompt_tokens, index=1)
txt = tokenizer.detokenize(output_tokens)
print(f"Greedy search generated text: \n{txt}\n")

sampler = keras_nlp.samplers.BeamSampler(num_beams=10)
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Beam search generated text: \n{txt}\n")

sampler = keras_nlp.samplers.RandomSampler()
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Random search generated text: \n{txt}\n")

sampler = keras_nlp.samplers.TopKSampler(k=10)
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Top-K search generated text: \n{txt}\n")

sampler = keras_nlp.samplers.TopPSampler(p=0.5)
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Top-P search generated text: \n{txt}\n")



