from pprint import pprint
import tensorflow as tf
from tensorflow import keras
import tensorflow_io as tfio
import keras_nlp

# db variables
URI = "mongodb://localhost:27017"
DATABASE = "smaller_jobs"
TRAIN_COLLECTION = "train_data"
VAL_COLLECTION = "val_data"

# data parameters
BATCH_SIZE = 32
SEQ_LEN = 128
MIN_TRAINING_SEQ_LEN = 300

# model params
EMBED_DIM = 256
FEED_FORWARD_DIM = 256
NUM_HEADS = 3
NUM_LAYERS = 2
VOCAB_SIZE = 6000

# training
EPOCHS = 30

# generating text
GEN_TOKENS = 100

# load in training and validation sets
pre_train_ds = tfio.experimental.mongodb.MongoDBIODataset(uri=URI, database=DATABASE, collection=TRAIN_COLLECTION)


SPECS = {"descr": tf.TensorSpec(tf.TensorShape([]), tf.string, name="descr_text"), }

# each line in dataset is a string that needs to be decoded into json, then take just the value. filter & batch
pre_train_ds = (
    pre_train_ds
    .map(lambda x: tfio.experimental.serialization.decode_json(x, specs=SPECS), num_parallel_calls=tf.data.AUTOTUNE,
         deterministic=False)
    .map(lambda x: x.values())
    .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
    .batch(BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer_size=256, seed=13)
)
# print(len(list(pre_train_ds.as_numpy_iterator())))
# data grouped into 1231 batches

pre_val_ds = tfio.experimental.mongodb.MongoDBIODataset(uri=URI, database=DATABASE, collection=VAL_COLLECTION)
pre_val_ds = (pre_val_ds
              .map(lambda x: tfio.experimental.serialization.decode_json(x, specs=SPECS),
                   num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
              .map(lambda x: x.values())
              .filter(lambda x: tf.strings.length(x) > MIN_TRAINING_SEQ_LEN)
              .batch(BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE)
              .shuffle(buffer_size=256, seed=13)
              )
val_d = list(pre_val_ds.as_numpy_iterator())


# train and load tokenizer
vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
    pre_train_ds,
    vocabulary_size=VOCAB_SIZE,
    reserved_tokens=["[PAD]", "[UNK]", "[BOS]"],
)

tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    sequence_length=SEQ_LEN,
)


start_packer = keras_nlp.layers.StartEndPacker(
    sequence_length=SEQ_LEN,
    start_value=tokenizer.token_to_id("[BOS]"),
)


def process(inputs):
    outputs = tokenizer(inputs)
    features = start_packer(outputs)
    labels = outputs
    return features, labels


# split into train and label sequences
train_ds = pre_train_ds.map(process, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds = pre_val_ds.map(process, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

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


def next(prompt, cache, index):
    logits = model(prompt)[:, index - 1, :]
    hidden_states = None
    return logits, hidden_states, cache


sampler = keras_nlp.samplers.GreedySampler()
output_tokens = sampler(next=next, prompt=prompt_tokens, index=1)
txt = tokenizer.detokenize(output_tokens)
print(f"Greedy search generated text: \n{txt}\n")
"""
[b'[BOS] < div > < p > < b > Job Description < / b > < / p > 
< p > < b > About Us : < / b > < / p > 
< p > < / p > 
< p > < b > Sainty Li Accomack Regional Sales Support Shncitation < / b > < / p > 
< p > < b > < br > The Office of the Chief 
Operating Officer ( CU ) team of 3D ) , and serves more effective professional providers < b > 1 . Shape planning and 
collaboration solutions to the mission to entitlement and educate students , first and foremost advocates to deliver 
transform their financial life through close coordination . This position is for : < / p > 
< p > < b > Chief Executive Officer < / b > < / p > 
< p > The Office Administrator oversees the Division of Finance , Executive Operations and Operations Officer']
"""

sampler = keras_nlp.samplers.BeamSampler(num_beams=10)
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Beam search generated text: \n{txt}\n")
"""
[b'[BOS] < div > < div > < div > < div > < div > < div > < div > < div > < div > < div > < div > < div > < div > < div >
 < div > < div > < div > < div > < div > < div > < div > < div > < div > < div > < div > < div > < div > < div > < div >
  < div > < div > < div > < div > < div > < div > < div > < div > < div > < div > < div > < div > < div > < div > < div 
  > < div > < div > < div > < div > < div > < div > < div > < div > < div > < div > < div > < div > < div > < div > < 
  div > < div > < div > < div > < div > < div']
"""

sampler = keras_nlp.samplers.RandomSampler()
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Random search generated text: \n{txt}\n")
"""
[b'[BOS] < p > T DOITIES & Spet SEC Specialist / NA - III F & Jun Clearance ( s ) : mission Secret security clearance < 
/ p > 
< p > DS : < / p > 
< p > : < / p >  
< p > 1 . 2 / Duperminder : 2S 3G , 5 : 30am . TAUSTOR CAPE Clearance Required : 30 : 00 < / p >  
< p > < / p > 
< p > < b > : < / b > No : No phone : 1 ) Staff Secret Principal Projects ( 2 . Inc . II / Unit : Develop Care Specialist < b > Bachelor \xe2\x80\x99 s Degree will apply . Master based security clearance while 2 : 30 or Master Schedule < / p > 
5 : Secret Informationisturgeant Weekly Security Officer Standards SCRINCILA < br > < / p > 
< p > 1 )']
"""

sampler = keras_nlp.samplers.TopKSampler(k=10)
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Top-K search generated text: \n{txt}\n")
"""
[b'[BOS] < p > < / p > 
< div > < div > < div > < div > < div > < div > DMALUST < / div > 
< div > < b > Role : POS is a UVA Community CRAva , CA , LLC ( NYSE < / b > < / div > 
< / div > 
< div > < i > Parsst rely known for by investing in  competitive salary and employees to employees and grown to minimize cost check - in our only community . < / b > < / div > 
< p > < b > Postdoctoral period a period of all levels : shower salary increases : starting pay range $ 1 . 00 a cream . 00 p . 00 p . hourly , but be temporary leave provided by appointment , benefits , retention , and facilitates  and staffing needs . . < / b > < / p > 
< p > Salary is commensurate with experience']
"""

sampler = keras_nlp.samplers.TopPSampler(p=0.5)
output_tokens = sampler(
    next=next,
    prompt=prompt_tokens,
    index=1,
)
txt = tokenizer.detokenize(output_tokens)
print(f"Top-P search generated text: \n{txt}\n")
"""
[b'[BOS] < div > < p > < b > POSITION SUMMARY : < / b > < / p > 
< p > Responsible for overseeing the entire day to day activities and document thirty orders for service departments . < / p >
 < p > < b > Reports To : < / b > < / p >
  < ul >
   < li > Assist in maintaining and product corrp and maintain the event goeseer control policies , procedures , inventory of the development of service and regulations of the set of hazardous waste management system ( retrubberies , and replacement . Animalony R & Dewards ) by the associated with the Reque , Shilver S . < / li > 
   < li > save , from flight meats and products < / li >
   < li > Making sure order sheet to dishwasher , pick - up apprentice , and w']
"""