# Task2
Task 2 is about reproducing the results of Fine Tuning BERT model.
In task 1 we trained BERT model for Question Answering. Here we are building a text classifier on CoLA(Corpus of Linguistic Acceptabilty) data and trying to fine tune to get results faster. 
Specifically, we will take the pre-trained BERT model, add an untrained layer of neurons on the end, and train the new model for our classification task. So what are the advantages of this approach.

1. **Quicker Development**: Since the pre-trained BERT model weights already encode a lot of information about the language, it takes much less time to train our fine-tuned model.  Acutally, the authors recommend only 2-4 epochs of training for fine-tuning BERT on a specific NLP task (compared to the hundreds of GPU hours needed to train the original BERT model or a LSTM from scratch!).
2. **Less Data** : Since we have pre-trained weights available, we can fine-tune our task on a much smaller dataset than would be required in a model that is built from scratch. By fine-tuning BERT, we now can do away with training a model to good performance on a much smaller amount of training data.
3. **Better Results**: Just by adding one fully-connected layer on top of BERT and training for a few epochs, we could achieve state of the art results with minimal task-specific adjustments for a wide variety of tasks.

### task 2 Training Log 
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
