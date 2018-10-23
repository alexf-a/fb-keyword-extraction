# Keyword Extraction
A keyword extraction algorithm for online questions. From Facebook Recruiting's Kaggle challenge: https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction

This is a multi-label classification problem. Training and testing data are both from Stack Exchange. 

## Strategy
I used TFIDF to encode posts and BOW to encode labels. This creates the multi-class multi-label classification problem of mapping input vectors to bit-vectors. I chose Complement Naive Bayes for the model, because of potential imbalances in the training set. 

To process the data, I removed duplicate measurements and made each label the union of all other labels that shared the same measurement. I also weighted titles more heavily in training.

The error metric chosen was the Hamming Loss. 
