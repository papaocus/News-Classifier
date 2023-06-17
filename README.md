# News-Classifier

Description:
This project focuses on developing a machine learning model for detecting fake news articles. With the increasing spread of misinformation and false information, it has become crucial to develop effective techniques to identify and combat fake news. The aim of this project is to leverage machine learning algorithms to automatically classify news articles as either real or fake based on their content.

The project begins with the preprocessing of the dataset, which includes cleaning the text by removing special characters, converting to lowercase, and applying stemming techniques to reduce the words to their root form. Stop words, such as common words that do not carry significant meaning, are also removed to improve the accuracy of the model.

Next, the dataset is split into training and testing sets using the train_test_split function from the sklearn.model_selection module. The training set is used to train a decision tree classifier and a logistic regression model. These classifiers are chosen for their ability to handle text classification tasks effectively.

To represent the text data numerically, the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique is applied using the TfidfVectorizer from the sklearn.feature_extraction.text module. This technique calculates the importance of each word in the documents and assigns higher weights to words that are more specific to a document and less frequent in the entire corpus.

The decision tree classifier and logistic regression model are trained using the TF-IDF vectorized features and their corresponding labels. The accuracy of the models is evaluated using the accuracy_score metric from the sklearn.metrics module.

The project also includes handling missing values in the dataset by filling them with empty strings. This ensures that all data points have valid content for further processing.

Finally, the trained models can be used to predict the authenticity of new news articles by transforming the input text into TF-IDF vectors and applying the trained models for classification.

The source code for this project is available on GitHub, providing a detailed implementation of the preprocessing steps, model training, evaluation, and prediction. This project serves as a valuable resource for researchers and developers interested in fake news detection using machine learning techniques.

Keywords: fake news detection, machine learning, text classification, decision tree classifier, logistic regression, TF-IDF vectorization, preprocessing, accuracy evaluation, GitHub.
