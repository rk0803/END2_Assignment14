# END2_Assignment14
Assignment 14 is on BERT and BART models. We were given three tasks.</br>
- TASK 1: Train BERT using the code based on Article by Michel Kana: https://towardsdatascience.com/bert-nlp-how-to-build-a-question-answering-bot-98b1d1594d7b, with 
Code credits to Michel Kana, and Prachur Bhargava, Lead Data Scientist @ Microsoft, on the Squad Dataset for 20% overall samples (1/5 Epochs) and to show output on 5 samples.
- TASK 2: is based on  BERT Fine-Tuning Tutorial with PyTorch, By Chris McCormick and Nick Ryan results, and show output on 5 samples.
- TASK 3: is based on the blog BART for Paraphrasing with Simple Transformers, by Thilina Rajpksha. We need to reproduce the training explained in this blog. We were allowed to pick fewer datasets
## Task1
### Plot of training loss
Here is a plot of the trend in the loss while training </br>
![image](https://user-images.githubusercontent.com/82941475/128807790-06f07deb-c8ab-474b-aab8-7149745cd779.png)

### Challenges faced
1. utils_squad and utils_squad_evaluate needed to looked for on the net and needed to be brought here.
2. Initially I took the complete dataset and decided to train on whole of it. My system crashed after 50% of the epoch due RAM being full. So changed the dataset size to 20% of the original dataset.
3. While evaluating the model, to get predictions, util_squad.write_predictions is giving error key_number 1000007200. There are 7205 samples in the validation dataset and each sample is given a unique_id starting from 1000000000. So technically I should have upto 1000007204 unique_ids. When I tried to investigate further, with a single batch (of size 16), it was found that it is trying to read key number 1000000016, whereas there are keys only upto 1000000015.  All this is availble in the notebook **task1_ass14_BERT_Tutorial_How_To_Build_a_Question_Answering_Bot.ipynb**

# Task2
