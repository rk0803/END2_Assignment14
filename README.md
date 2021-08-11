# END2_Assignment14
Assignment 14 is on BERT and BART models. We were given three tasks.</br>
- TASK 1: Train BERT using the code based on Article by Michel Kana: https://towardsdatascience.com/bert-nlp-how-to-build-a-question-answering-bot-98b1d1594d7b, with 
Code credits to Michel Kana, and Prachur Bhargava, Lead Data Scientist @ Microsoft, on the Squad Dataset for 20% overall samples (1/5 Epochs) and to show output on 5 samples.
- TASK 2: is based on  BERT Fine-Tuning Tutorial with PyTorch, By Chris McCormick and Nick Ryan results, and show output on 5 samples.
- TASK 3: is based on the blog BART for Paraphrasing with Simple Transformers, by Thilina Rajpksha. We need to reproduce the training explained in this blog. We were allowed to pick fewer datasets
## Task1
This task is on building a question answering bot using NLP. In these type tasks, the model receives a question regarding text content and is required to mark the beginning and end of the answer in the text. A neural net can be trained to learn relationships between context, questions, and answers bringing a large set of such texts together with sample questions and the position of the answers in the text. This is analogous to the reading comprehension, where a passage is given and questions are framed based on the passage. The plausibility of building this kind of bots has been strengthened with the availability of large labeled datasets. </br>
The Stanford Question Answering Dataset (SQuAD) is one such example of large-scale labeled datasets for reading comprehension. Rajpurkar et al. developed SQuAD 2.0, which combines 100,000 answerable questions with 50,000 unanswerable questions about the same paragraph from a set of Wikipedia articles. The unanswerable questions were written adversarially by crowd workers to look similar to answerable ones. 
To understand the task, we first try to understand the dataset followed by the model BERT.
#### Here is snapshot of squad dataset. </br>
qas_id	| question_text |	doc_tokens	| orig_answer_text	| start_position	| end_position	| is_impossible 
--------|---------------|-------------|-------------------|-----------------|---------------|---------
0	| When did Beyonce start becoming popular?	| \[Beyoncé, Giselle, Knowles-Carter, (/biːˈjɒnse...	| in the late 1990s|	39	| 42	| False
1	| What areas did Beyonce compete in when she was... |\[Beyoncé, Giselle, Knowles-Carter, (/biːˈjɒnse...| singing and dancing	| 28	| 30	|False
2	|  When did Beyonce leave Destiny's Child and bec...|	\[Beyoncé, Giselle, Knowles-Carter, (/biːˈjɒnse... |	2003	| 82	| 82	| False
3	| In what city and state did Beyonce grow up?	 | \[Beyoncé, Giselle, Knowles-Carter, (/biːˈjɒnse...	| Houston, Texas	| 22 |	23	| False
4 |	In which decade did Beyonce become famous? 	| \[Beyoncé, Giselle, Knowles-Carter, (/biːˈjɒnse...	| late 1990s	| 41	| 42	| False

Here most of the column names are self explanatory while some need little explanation.
- *doc_tokens* describes the context, i.e. the text which we want our model to understand.
- The answer is always a portion from the context starting at *start_position* and ending at *end_position*. 
- If the question does not have any answer in the context, *is_impossible* has the value true.
## BUT What is BERT?
**BERT** is Google’s  NLP framework, and seemingly the most influential one in recent times. </br>
**"BERT stands for Bidirectional Encoder Representations from Transformers. It is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of NLP tasks.”**
#### Salient features:
- BERT is a trained Transformer Encoder stack, with twelve in the Base version, and twenty-four in the Large version. 
- BERT was trained on Wikipedia and Book Corpus, a dataset containing +10,000 books of different genres. 
- BERT is a bidirectional model, i.e. earns information from both the left and the right side of a token’s context during the training phase. This important for meaningful understanding of the language.
- We can fine-tune it by adding just a couple of additional output layers to create state-of-the-art models for a variety of NLP tasks.
#### Need for a BERT like model
Initially, learning language representions using word embeddings like Word2Vec and GloVe, changed the way we performed NLP tasks. We could capture contextual relationships among words. But there was a limit to the amount of information they could capture and this motivated the use of deeper and more complex language models, like layers of LSTMs and GRUs. Also, these models did not take context of the word into account i.e. same word, if it has different meanings in different contexts, these embedding would give the same vector even for different contexts.
To handle this loss of valuable information, new models like **Embeddings from Language Models** (ELMo) and **Universal Language Model Fine-tuning** (ULMFiT) paved the way for transfer learning. i.e. **Transfer Learning in NLP = Pre-Training and Fine-Tuning**
Methods of pre-training and fine-tuning, introduced by ULMFiT and ELMo were extended by **OpenAI’s GPT**replacing the LSTM-based architecture for Language Modeling with a Transformer-based architecture. These GPT-based models could be fine-tuned to multiple NLP tasks beyond document classification, such as common sense reasoning, semantic similarity, and reading comprehension. Essentially, the emphasis was on Transformer framework, which can learn complex patterns in the data by using the Attention mechanism and can train faster than an LSTM-based model.
### Diving into BERT
With the dawn of transfer learning, solving NLP tasks became a 2-step process:
1. the Train a language model on a large unlabelled text corpus (unsupervised or semi-supervised)
2. Fine-tune this large model to specific NLP tasks to utilize the large repository of knowledge this model has gained (supervised)
#### BERT’s Architecture
The BERT architecture builds on top of Transformer. We currently have two variants available:

- BERT Base: 12 layers (transformer blocks), 12 attention heads, and 110 million parameters
- BERT Large: 24 layers (transformer blocks), 16 attention heads and, 340 million parameters
![image](https://user-images.githubusercontent.com/82941475/128996342-5c5bb7ce-167a-43eb-ae13-7f643ae88d79.png)
#### Text pre-processing in BERT
BERT has a specific set of rules to represent the input text for the model. Many of these are creative design choices that make the model even better.
![image](https://user-images.githubusercontent.com/82941475/128996582-991fbfc3-0a24-4a4f-a897-3e7829901bdc.png)

Every input embedding is a combination of 3 embeddings:
1. **Position Embeddings**: BERT learns and uses positional embeddings to express the position of words in a sentence. These are added to overcome the limitation of Transformer which, unlike an RNN, is not able to capture “sequence” or “order” information
2. **Segment Embeddings**: BERT can also take sentence pairs as inputs for tasks (Question-Answering). That’s why it learns a unique embedding for the first and the second sentences to help the model distinguish between them. In the above example, all the tokens marked as EA belong to sentence A (and similarly for EB)
3. **Token Embeddings**: These are the embeddings learned for the specific token from the WordPiece token vocabulary

#### Pre-training Tasks
BERT is pre-trained on Masked Language Modeling and Next Sentence Prediction. 
The two images below explain well about these two task.
**Masked Language Modeling** </br>
![image](https://user-images.githubusercontent.com/82941475/128998507-546cf4ee-f40b-4bca-96fb-e489c2d7b5d7.png)

**Next Sentence Prediction** </br>
![image](https://user-images.githubusercontent.com/82941475/128998464-cdccff4c-c260-46fc-a845-b0bee6348442.png)

So now that BERT has opened up enormous opportunities, we can take advantage of BERT’s large repository of knowledge for our NLP applications in multiple ways. Most common and powerful one is to fine-tune it on your own task and task-specific data using embeddings from BERT as embeddings for our text documents.

In this section, we will learn how to use BERT’s embeddings for our NLP task. We’ll take up the concept of fine-tuning an entire BERT model in one of the future articles.

## So TASK 1- Transfer Learning using BERT for Question Answering 
Here BERT is used to extract high-quality language features from the SQuAD text by adding a single linear layer on top. This  linear layer has two outputs, the first for predicting the probability that the current subtoken is the start of the answer and the second output for the end position of the answer.

#### Plot of training loss
Here is a plot of the trend in the loss while training </br>
![image](https://user-images.githubusercontent.com/82941475/128807790-06f07deb-c8ab-474b-aab8-7149745cd779.png)

#### Challenges faced
1. utils_squad and utils_squad_evaluate needed to looked for on the net and needed to be brought here.
2. Initially I took the complete dataset and decided to train on whole of it. My system crashed after 50% of the epoch due RAM being full. So changed the dataset size to 20% of the original dataset.
3. While evaluating the model, to get predictions, util_squad.write_predictions is giving error key_number 1000007200. There are 7205 samples in the validation dataset and each sample is given a unique_id starting from 1000000000. So technically I should have upto 1000007204 unique_ids. When I tried to investigate further, with a single batch (of size 16), it was found that it is trying to read key number 1000000016, whereas there are keys only upto 1000000015.  All this is availble in the notebook **task1_ass14_BERT_Tutorial_How_To_Build_a_Question_Answering_Bot.ipynb**

# Task2
