# END2_Assignment14
Assignment 14 is on BERT and BART models. We were given three tasks.</br>
- TASK 1: Train BERT using the code based on Article by Michel Kana: https://towardsdatascience.com/bert-nlp-how-to-build-a-question-answering-bot-98b1d1594d7b, with 
Code credits to Michel Kana, and Prachur Bhargava, Lead Data Scientist @ Microsoft, on the Squad Dataset for 20% overall samples (1/5 Epochs) and to show output on 5 samples.
- TASK 2: is based on  BERT Fine-Tuning Tutorial with PyTorch, By Chris McCormick and Nick Ryan results, and show output on 5 samples.
- TASK 3: is based on the blog BART for Paraphrasing with Simple Transformers, by Thilina Rajpksha. We need to reproduce the training explained in this blog. We were allowed to pick fewer datasets
## Task1
### Challenges faced
1. utils_squad and utils_squad_evaluate needed to looked for on the net and needed to be brought here.
2. Initially I took the complete dataset and decided to train on whole of it. My system crashed after 50% of the epoch due RAM being full.
