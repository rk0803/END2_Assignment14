# Task2
Task 2 is about reproducing the results of Fine Tuning BERT model.
In task 1 we trained BERT model for Question Answering. Here we are building a text classifier on CoLA(Corpus of Linguistic Acceptabilty) data and trying to fine tune to get results faster. 
Specifically, we will take the pre-trained BERT model, add an untrained layer of neurons on the end, and train the new model for our classification task. So what are the advantages of this approach.
### Advantages of Using pre-trained BERT model
1. **Quicker Development**: Since the pre-trained BERT model weights already encode a lot of information about the language, it takes much less time to train our fine-tuned model.  Acutally, the authors recommend only 2-4 epochs of training for fine-tuning BERT on a specific NLP task (compared to the hundreds of GPU hours needed to train the original BERT model or a LSTM from scratch!).
2. **Less Data** : Since we have pre-trained weights available, we can fine-tune our task on a much smaller dataset than would be required in a model that is built from scratch. By fine-tuning BERT, we now can do away with training a model to good performance on a much smaller amount of training data.
3. **Better Results**: Just by adding one fully-connected layer on top of BERT and training for a few epochs, we could achieve state of the art results with minimal task-specific adjustments for a wide variety of tasks.

### BERT approach to classification
#### Special Tokens

1)**`[SEP]`** and  **`[CLS]`**

At the end of every sentence,  special `[SEP]` token is appended. This token is an artifact of two-sentence tasks, where BERT is given two separate sentences and asked to determine something (e.g., can the answer to the question in sentence A be found in sentence B?). 

For classification tasks, a special `[CLS]` token is prepended to the beginning of every sentence. This token has special significance. BERT consists of 12 Transformer layers. Each transformer takes in a list of token embeddings, and produces the same number of embeddings on the output, with changed feature values changed. 
![Illustration of CLS token purpose](https://drive.google.com/uc?export=view&id=1ck4mvGkznVJfW3hv6GUqcdGepVTOx7HE)

**On the output of the final (12th) transformer, *only the first embedding (corresponding to the [CLS] token) is used by the classifier*.**

#### Sentence Length & Attention Mask
How does BERT handle varying length of sentences in the dataset?

BERT has two constraints:
1. All sentences must be padded or truncated to a single, fixed length.
2. The maximum sentence length is 512 tokens.

Padding is done with a special `[PAD]` token, which is at index 0 in the BERT vocabulary. The below illustration demonstrates padding out to a "MAX_LEN" of 8 tokens.

<img src="https://drive.google.com/uc?export=view&id=1cb5xeqLu_5vPOgs3eRnail2Y00Fl2pCo" width="600">

The "Attention Mask" is simply an array of 1s and 0s indicating which tokens are padding and which aren't (seems kind of redundant, doesn't it?!). This mask tells the "Self-Attention" mechanism in BERT not to incorporate these PAD tokens into its interpretation of the sentence.

### Training log snippet
======== Epoch 3 / 4 ========
Training...
  Batch    40  of    241.    Elapsed: 0:00:15.
  Batch    80  of    241.    Elapsed: 0:00:30.
  Batch   120  of    241.    Elapsed: 0:00:45.
  Batch   160  of    241.    Elapsed: 0:01:01.
  Batch   200  of    241.    Elapsed: 0:01:16.
  Batch   240  of    241.    Elapsed: 0:01:32.

  Average training loss: 0.18
  Training epcoh took: 0:01:32

Running Validation...
  Accuracy: 0.81
  Validation Loss: 0.60
  Validation took: 0:00:04

======== Epoch 4 / 4 ========
Training...
  Batch    40  of    241.    Elapsed: 0:00:15.
  Batch    80  of    241.    Elapsed: 0:00:31.
  Batch   120  of    241.    Elapsed: 0:00:46.
  Batch   160  of    241.    Elapsed: 0:01:02.
  Batch   200  of    241.    Elapsed: 0:01:17.
  Batch   240  of    241.    Elapsed: 0:01:32.

  Average training loss: 0.13
  Training epcoh took: 0:01:33

Running Validation...
  Accuracy: 0.81
  Validation Loss: 0.73
  Validation took: 0:00:04

Training complete!
Total training took 0:06:12 (h:mm:ss)

### Training Log 
epoch | Training Loss	 | Valid. Loss	| Valid. Accur.	| Training Time	| Validation Time
-----|-----------------|--------------|---------------|---------------|-----------------				
1 |	0.51	| 0.46	| 0.81	| 0:02:38	| 0:00:06
2	| 0.31	| 0.46	| 0.81	| 0:02:38	| 0:00:06
3	| 0.20	| 0.55	| 0.82	| 0:02:38	| 0:00:06
4	| 0.13	| 0.65	| 0.82	| 0:02:38	| 0:00:06

### MCC Score
Accuracy on the CoLA benchmark is measured using the "Matthews correlation coefficient" (MCC).
![image](https://user-images.githubusercontent.com/82941475/129001269-3b95d61e-08a0-451d-8417-d04d2067a65c.png)

The final MCC score came out to be **Total MCC: 0.571**

### Sample output on 5 sentences
Sentence | Predicted Label | Actual Label
------------|------------|---------------
Somebody just left - guess who. |   1 | 1
She knew French for Tom. |  0 | 0
John is tall on several occasions. |  0 | 0
Everyone relies on someone. It's unclear who. |  1 |  1
They noticed the painting, but I don't know for how long. |  0 |  0
