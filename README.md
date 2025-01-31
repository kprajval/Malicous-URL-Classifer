Malicious URL Classifier
Project Overview
The Malicious URL Classifier aims to categorize URLs into four distinct classes to help identify and mitigate online security risks:

Benign: Safe, legitimate URLs.
Defacement: URLs linked to website defacements.
Phishing: URLs used for deceptive practices, often aiming to steal user credentials.
Malware: URLs hosting malicious software.
This project employs a variety of text classification models to analyze the patterns within URLs and classify them based on their learned features, offering an efficient way to detect harmful URLs.

Models Used and Performance Analysis
Several machine learning and deep learning models were implemented and evaluated based on their accuracy in classifying URLs. Below are the results of the models tested:

1. Naïve Bayes – 86% Accuracy
Description: A probabilistic model based on Bayes' theorem, assuming feature independence.

3. Logistic Regression – 91% Accuracy
Description: A linear model that estimates probabilities using the logistic function.

5. BERT Classifier – 97.89% Accuracy
Description: A transformer-based deep learning model known for its ability to understand context and semantics in text.

7. LSTM (Long Short-Term Memory) – 96.02% Accuracy
Description: A recurrent neural network (RNN) model designed for processing sequential data.

9. RNN + CNN Model – 96.57%
Description: A hybrid model combining the strengths of both Recurrent Neural Networks (RNN) and Convolutional Neural Networks (CNN).

Conclusion
The BERT Classifier demonstrated superior performance with an accuracy of 97.89%, making it the most effective model for malicious URL classification in this project. Its ability to understand context and semantics within URLs played a crucial role in its success. However, for more lightweight implementations, the Naïve Bayes model offers a reliable alternative with good accuracy and computational efficiency.
