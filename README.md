# Machine_Learning
My machine learning projects. It's time to dive deep into the machine learning libraries,
and study the models, the hyperparameters, scaling, and everything relates to this topic.

All the notebooks, I try to comment as much as possible, to make them clear, understandable, and reproducible to anyone. 

## Pump it up!
It's a DrivenData challenge, you have to do a multiclass classification to predict African water pump conditions, if they work, 
don't work, or work but repair is needed. Currently there are 8000+ competitors, and the best score is around 82%. My solution is currently the 1640th with 75.25% accuracy. My ideas to make it better is on the bottom of the notebook, and I update it regularly.

## Image classification
It's a Deep Learning project, the MNIST dataset of the hand written numbers. At the end of the notebook, I added some examples
of the predictions and the real labels - for this part, the real judge are your eyes.

## Musical recommender 
There are two of them. One is the full version of a musical recommender system. We have two datasets, one is around 2900 rows with
the users, the play count, and the artist's id. The other contains the artist names. I learned to use pandas pivot function, the 
csr_matrix, and the NMF (Non-negative Matrix Factorization). You choose an artist from the given list, and the result is a list 
of the recommended artists based on the user play counts.

There's another version of this notebook, the "WS" one, which is made for the budapest.py workshop. It's better to present it with
some deleted pieces, because with the attendees, we have to do it ourselves together in order to make it work.

## Titanic
This was the first machine learning project I tried. I learned a lot while solving it, because I tried every possible version I could think. And because it's a Kaggle competition, - which is closed by now, but you can still send your prediction to score - I uploaded more than 60 times to see if it's improved. I learned about classification in general, how to evaluate the model, how to split it. What is Feature Engineering, how can I make from two "maybe useless" columns one useful. I learned about Plotly Express and Seaborn for visualization. I tried Decision Trees, Random Forests, and how to tune them with Hyperparameter Tuning. Cross Validation was another new thing for me. My score was about 75% which I thought was not bad, but could be better. That's when I found DataCamp's course about XGBoost, and after completing it, I tried it, and I got above 80% score. It made me real proud, it was a good project.

## Wine dataset unsupervised learning
I got the idea from a DataCamp course, where it was briefly introduced. The wine dataset is a Multi Class Classification problem, where you have three categories of wines, and a lot of features describing them, and you have to tell, which wine is it, from just the features. This task was modified for Unsupervised Learning, so I removed the labels, and tried my first clustering model, KMeans, with three components. The main idea of this that you have to scale your features, because my first results were not promising. KMeans couldn't cluster two of the three wines steadily. After StandardScaling the problem was solved, because it handled the variance.

## Data_Cleaning_bppy notebook 
This is the complete notebook about data cleaning with Pandas. We have two examples, one is the gapminder dataset, which holds the life expectancy of every countries from 1800 to 2016. This is a messy, and untidy dataset originally, but with Pandas' data cleaning functions we clean it, and tidy it. The second example is the tips dataset, intentionally ruined, so we can clean it. It holds more than two hundred rows, with bills, tips, day of the dinner or lunch, and so on. We used pandas' to_numeric, get_dummies, drop, replace functions among the "usual" ones. LabelEncoding and One-Hot Encoding was also introduced. What is the difference between the two, and when to use which (with wrong examples to make it more clear).

## Scikit-learn classification task
The bank notes dataset was the data, from four features we have to predict if a bank note is authentic, or inauthentic. It's a binary classification example, with a preprocessed dataset, so the focus could be on the models. I used five popular models, Logistic Regression, Decision Tree, Random Forest, Support Vector Machine and XGBoost. Other than the models, Grid Search, and Cross Validation was used too. The evaluation metric was the confusion matrix, and the classification report. See the results in the notebook.
