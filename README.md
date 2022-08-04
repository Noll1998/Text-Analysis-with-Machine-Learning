# Text-Analysis-with-Machine-Learning
Applying NLP techniques to analyze a text dataset with Python libraries and tools


  This project uses the Python libraries NumPy, Matplotlib, Pandas, Sklearn, Seaborn, and NLTK to do text analysis with Machine Learning methods. The techniques applied are text preprocessing such as stemming and lemmatization, as well as dimensionality reduction with t-SNE.  
  Natural Language Process (NLP) applies Machine Learning techniques to analyze text and comprehend its meaning. One of the most popular datasets for text analytics is the 20 Newsgroups Dataset, which is a collect of approximately 20,000 documents about 20 different topics, ranging from electronics/hardware to sports and cars. 
The data is in the form of a dictionary and its keys are as follows:
<img src="https://user-images.githubusercontent.com/68149933/182965521-050374db-fac9-4422-bb85-24107a3faec3.png" width="500">

  The ‘data’ key holds the content of the documents while ‘filenames’ contains the path to the document within the filesystem. The ‘target_names’ key is the category to which the document belongs to, such as ‘alt.atheism’, ‘comp.graphics’, and ‘comp.os’. The ‘target’ key shows a number between 0 and 19, corresponding to the key to one of the 20 available categories. As shown below, the distribution of the topics is symmetric, therefore each category is fairly represented compared to the others. 
  
  <img src="https://user-images.githubusercontent.com/68149933/182965727-46b96d35-f11b-4358-81dd-4519693ad9e4.png" width="300">

  The Bag of Words (BoW) Model considers the count of how many times a word appears and does not account for grammatical structure or word order. For this dataset, the most recurrent 500 items are:  

<img src="https://user-images.githubusercontent.com/68149933/182965780-d9d07771-8d8b-40d5-aab1-90a2ec07c8de.png" width="580">

  It’s noticeable that some elements in this matrix are not useful for the analysis, such as the numbers, the mix of letters with numbers (a86) and very common words like ‘by’, ‘as’, and ‘at’. After keeping letter-only items, excluding stop words (the very common words just mentioned), stemming (chopping off the ends of words so ‘girls’ and ‘girl’ can be counted as one key) and lemmatizing (a more complex form of stemming, it takes a word back to its root form as in ‘saw’ becoming ‘see’) the dataset, we get the following result: 
  
<img src= "https://user-images.githubusercontent.com/68149933/182965883-8a153f69-eb0c-41b0-ab0c-f204dbd41883.png" width="400">
<img src= "https://user-images.githubusercontent.com/68149933/182965900-99940d2d-8f6f-4195-b787-195683a942f9.png" width="400">

  Since this vector has 500 features, it is necessary to perform a dimensionality reduction to get a more clarity from the data. The dimensionality reduction method is useful because it retains the information while transforming into a low-dimensional space. With the data in 2-D, by choosing 5 related topics from the 20 available with the t-SNE method, the result is the following scatter plot:

<img src= "https://user-images.githubusercontent.com/68149933/182966052-38eeeca1-f66c-4a36-8d84-8275f63d2190.png" width="320">

  In this plot, the data points that are closer together represent the same topic, while data points spread apart are less related topic-wise. The t-SNE method used stands for t-Distributed Stochastic Neighbor Embedding which is an unsupervised technique used for visualizing high-dimensional data.
