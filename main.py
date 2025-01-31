import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, SimpleRNN, Dense, SpatialDropout1D, Conv1D, MaxPooling1D, Flatten
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

def preProcessing(datasetFilePath):
    df = pd.read_csv(datasetFilePath)
    df.dropna(inplace=True)
    y = df["type"]
    X = df.drop("type", axis=1)
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X['url'], y, test_size=0.42, random_state=42)
    return X_train, X_test, y_train, y_test

def LogisticRegressionMethod(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))

def NaiveBayesMethod(X_train, X_test, y_train, y_test):
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))

def LSTMMethod(X_train, X_test, y_train, y_test):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=100)
    X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=100)
    model = Sequential([
        Embedding(input_dim=5000, output_dim=128, input_length=100),
        SpatialDropout1D(0.2),
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        Dense(4, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test))
    y_pred = model.predict_classes(X_test_pad)
    print(classification_report(y_test, y_pred))

def CNN_RNNMethod(X_train, X_test, y_train, y_test):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=100)
    X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=100)
    
    model = Sequential([
        Embedding(input_dim=5000, output_dim=128, input_length=100),
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        SimpleRNN(100, activation='relu'),
        Dense(4, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_data=(X_test_pad, y_test))
    y_pred = model.predict_classes(X_test_pad)
    print(classification_report(y_test, y_pred))

def BERTMethod(X_train, X_test, y_train, y_test, batch_size=100, epochs=3, learning_rate=2e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_encodings = tokenizer(X_train.to_list(), padding=True, truncation=True, max_length=128, return_tensors="pt")
    test_encodings = tokenizer(X_test.to_list(), padding=True, truncation=True, max_length=128, return_tensors="pt")
    train_labels = torch.tensor(y_train, dtype=torch.long)
    test_labels = torch.tensor(y_test, dtype=torch.long)
    train_dataset = TensorDataset(train_encodings.input_ids, train_encodings.attention_mask, train_labels)
    test_dataset = TensorDataset(test_encodings.input_ids, test_encodings.attention_mask, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    num_labels = len(set(y_train))
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
            b_input_ids, b_attention_mask, b_labels = [item.to(device) for item in batch]
            optimizer.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_attention_mask, labels=b_labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}: Training Loss = {total_train_loss / len(train_loader):.4f}")

def main():
    dataset = "malicious_phish1.csv"
    X_train, X_test, y_train, y_test = preProcessing(dataset)
    #LogisticRegressionMethod(X_train, X_test, y_train, y_test)
    #NaiveBayesMethod(X_train, X_test, y_train, y_test)
    #LSTMMethod(X_train, X_test, y_train, y_test)
    CNN_RNNMethod(X_train, X_test, y_train, y_test)
    #BERTMethod(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
