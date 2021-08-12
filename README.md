# END2_Assignment14
Assignment 14 is on BERT and BART models. We were given three tasks.</br>
- TASK 1: Train BERT using the code based on Article by Michel Kana: https://towardsdatascience.com/bert-nlp-how-to-build-a-question-answering-bot-98b1d1594d7b, with 
Code credits to Michel Kana, and Prachur Bhargava, Lead Data Scientist @ Microsoft, on the Squad Dataset for 20% overall samples (1/5 Epochs) and to show output on 5 samples.
- TASK 2: is based on  BERT Fine-Tuning Tutorial with PyTorch, By Chris McCormick and Nick Ryan results, and show output on 5 samples.
- TASK 3: is based on the blog BART for Paraphrasing with Simple Transformers, by Thilina Rajpksha. We need to reproduce the training explained in this blog. We were allowed to pick fewer datasets



# Task 3
Task 3 is BART for Paraphrasing with Simple Transformers. Lets understand step by step.
### BART
BART is a denoising autoencoder for pretraining sequence-to-sequence models. BART is trained by (1) corrupting text with an arbitrary noising function, and (2) learning a model to reconstruct the original text.

BART uses a standard Transformer architecture (Encoder-Decoder) and is a combination of BERT, which is only encoder-model and GPT, which is a decoder-only model.

### Pre-Training BART
BART is pre-trained by minimizing the cross-entropy loss between the decoder output and the original sequence.

#### Masked Language Modeling (MLM), using BERT ( as discussed in task1)
MLM models such as BERT are pre-trained to predict masked tokens. i.e. Replace a random subset of the input with a mask token [MASK], which can said be as Adding noise/corruption, then original tokens for each of the [MASK] tokens, which can be called Denoising.

Importantly, because BERT models can “see” the the tokens before and after the masked tokens, when attempting to predict the original tokens, BERT is a bidirectional model.

This is suitable for classification tasks, information from the full sequence is needd to perform the prediction. However, for text generation tasks where the prediction depends only on the previous words, it is not suitable.

#### Autoregressive Models
Models which use previous inputs, to predict the next token are said to be autoregressive, such as GPT2, which are pre-trained to predict the next token given the previous sequence of tokens. Since they can't see the full sentence, they are not much suitable for classification

#### BART Sequence-to-Sequence
BART has both an encoder (like BERT) and a decoder (like GPT). The encoder uses a denoising objective similar to BERT while the decoder attempts to reproduce the original sequence (autoencoder), token by token, using the previous (uncorrupted) tokens and the output from the encoder. This gives multiple ways to add noise to the text. The corruption schemes used in the paper are summarized below.

Name | Description | Example 
-----|---------|---------
Token Masking | A random subset of the input is replaced with [MASK] tokens, like in BERT. | **ABC.DE.** changed to **A_C._E.**, 	Both **B** and **D** are masked with a single mask token for each.
Token Deletion | Random tokens are deleted from the input. The model be able to must decide which positions are missing | **ABC.DE.** is changed to 	**A.C.E.**	Both **B** and **D** are deleted and not replaced. 
Text Infilling | A number of text spans (length can vary) are each replaced with a single [MASK] token.| **ABC.DE.** is changed to 	**A_.D_E.** The span **BC** is replaced with a single mask token. A 0 length span is inserted between **D** and **E**. 
Sentence Permutation | The input is split based on periods (.), and the sentences are shuffled.| **ABC.DE.** is changed to 	**DE.ABC. **
Document Rotation |  A token is chosen at random, and the sequence is rotated so that it starts with the chosen token. |**ABC.DE.**	is changed to **C.DE.AB**	The sequence is rotated around C. 

The authors note that training BART with text infilling yields the most consistently strong performance across many tasks.

The task we are interested in, i.e. paraphrasing, the pre-trained BART model can be fine-tuned directly using the input sequence (original phrase) and the target sequence (paraphrased sentence) as a Sequence-to-Sequence model.

This also works for tasks like summarization and abstractive question answering.

