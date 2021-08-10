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
#### Little bit about BERT
BERT is a trained Transformer Encoder stack, with twelve in the Base version, and twenty-four in the Large version. BERT was trained on Wikipedia and Book Corpus, a dataset containing +10,000 books of different genres. More architectural details are available online, and can be referred to if required.

#### Transfer Learning using BERT for Question Answering 
Here BERT is used to extract high-quality language features from the SQuAD text by adding a single linear layer on top. This  linear layer has two outputs, the first for predicting the probability that the current subtoken is the start of the answer and the second output for the end position of the answer.

#### Plot of training loss
Here is a plot of the trend in the loss while training </br>
![image](https://user-images.githubusercontent.com/82941475/128807790-06f07deb-c8ab-474b-aab8-7149745cd779.png)

#### Challenges faced
1. utils_squad and utils_squad_evaluate needed to looked for on the net and needed to be brought here.
2. Initially I took the complete dataset and decided to train on whole of it. My system crashed after 50% of the epoch due RAM being full. So changed the dataset size to 20% of the original dataset.
3. While evaluating the model, to get predictions, util_squad.write_predictions is giving error key_number 1000007200. There are 7205 samples in the validation dataset and each sample is given a unique_id starting from 1000000000. So technically I should have upto 1000007204 unique_ids. When I tried to investigate further, with a single batch (of size 16), it was found that it is trying to read key number 1000000016, whereas there are keys only upto 1000000015.  All this is availble in the notebook **task1_ass14_BERT_Tutorial_How_To_Build_a_Question_Answering_Bot.ipynb**

# Task2
