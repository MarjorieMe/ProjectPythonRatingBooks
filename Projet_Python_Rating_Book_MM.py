#!/usr/bin/env python
# coding: utf-8

# # Projet Python : Predicts a book’s rating

# The project consists of predicting the coast of a book through a dataset. 
# For this, we must use the "machine learning" system. This makes it possible to apply predictive analysis algorithms to different types of data in order to predict the future. 

# In python, we need to import several libraries:

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# To meet our need, we have to go through different steps: 
#    - Data analysis: data processing, data cleaning, exploratory analysis and relevant graphics
#    - "Feature selection": Engineering of characteristics, pruning of characteristics and justification of choices;
#    - "Model Training": Model selection and justification and comparison with other models;
#    - "Model evaluation": Measurement and interpretation of results;
#    - Project Report

# ## Import Data with CSV

# First, we have to import the books.csv file into a dataframe using the `read_csv` function of the `pandas` library.

# In[2]:


dfBook = pd.read_csv("books.csv",sep=",",index_col="bookID", skipinitialspace=True, on_bad_lines='warn')


# > Remark: There is a data problem in the csv file due to empty fields or poorly constructed fields. To solve this problem, we must specify that lines that encounter a problem must trigger a warning and move on to the next one. 
# 
# > Remark: The first parameter corresponds to the name of our file that we want to import. The sep parameter specifies the delimiter to use, in our case, it is a comma that separates the different fields in the file. The `index_col`parameter corresponds to the column to be used as the row labels of the `DataFrame`, this corresponds to an id most of the time. The optional parameter `skipinitialspace` allows you to remove spaces after a delimiter, this one waits for a boolean (TRUE/FALSE), we noticed that in the dataset, there were spaces after the comma and this was a problem in the structure of the `DataFrame`.
# 

# Selection of the first 30 lines:

# In[3]:


dfBook.head(30)


# Selection of the last 10 lines:

# In[4]:


dfBook.tail(10)


# Selecting a 10 lines random dataset:

# In[5]:


dfBook.sample(10)


# We get the information from our dataframe `dfBook` through the `info()` function of the `pandas` library.

# In[6]:


dfBook.info()


# In order to better understand our dataset, we can also use the `describe()` function, which displays basic statistical details such as the mean, minimum, maximum, ... columns.

# > Remark: The `describe()` function returns the different statistical details only on the numerical columns (float, int, ...)

# In[7]:


dfBook.describe()


# We can also use the `shape` tuple to just look at the number of rows and columns in our dataframe: 

# In[8]:


dfBook.shape


# > Remark: The first digit corresponds to the rows and the second digit corresponds to the columns.

# # Data Cleaning, Exploratory Analysis and Relevant Charts

# In order to have a more concrete analysis process, we need to clean the data to make our data set more "clean" and closer to reality. This process allows you to modify or delete incorrect, incomplete, irrelevant, corrupted, duplicated or formatted data.

# To do this, we need to look more closely at our data:

# ## Nulls

# Using the `isna()` and `sum()` function, we can see the number of null values (specified by `NA`) in each column of the dataframe:

# In[9]:


dfBook.isna().sum()


# > Conclusion: We can see that we do not have null data.

# ## The empty values

# In order to better locate empty values, we must, using the function `replace()`, replace empty values (specified by `''`) by values `na` (specified by `np.nan`):

# In[10]:


dfBook.replace('', np.nan)
dfBook.isna().sum()


# > Conclusion: We can also see that there are no empty values.

# ## The columns "useless"

# This step depends on which column we want to analyze. For our part, we delete the isbn13, isbn and publisher column using the `drop()` function. These columns will not be useful after our analysis. As long as the 'publisher' column could be analyzed.

# In[11]:


dfBook.shape


# In[12]:


dfBookClear = dfBook.drop(columns=["isbn13", "isbn", "publisher"])


# We look to see if the column number has decreased.

# In[13]:


dfBookClear.shape


# We can also check with the `info()` function:

# In[14]:


dfBookClear.info()


# Or with the `describe()` function:

# In[15]:


dfBookClear.describe()


# ## Removing duplicates from title, language and authors

# In a large dataset like this, most of the time we have duplicates. To locate them, we need to copy the initial dataframe into a temporary dataframe and display the lines that are duplicated and then compare the results. 
# 
# In our case, we will say that a duplicate is a book that has title, the same author and the same language.

# In[16]:


dfDuplicatesRows = dfBookClear.loc[dfBookClear.sort_values('ratings_count').duplicated(subset=['title','authors','language_code'], keep='last')]
dfDuplicatesRows


# For example for the book "Treasure Island":

# In[17]:


dfBookClear.loc[(dfBookClear['title'] == "Treasure Island") & (dfBookClear['language_code'] == "eng") & (dfBookClear['authors'] ==  "Robert Louis Stevenson")]


# > Remark: This step consists in verifying the duplicates. 

# We want to keep the book that has more votes so that the analysis is more reliable.
# To do this, we sort the dataset in ascending order (smaller to larger) according to the number of votes and then delete the first (lower).

# In[18]:


dfBookClear = dfBookClear.drop_duplicates(subset=['title','authors','language_code'])
dfBookClear


# In[19]:


dfBookClear.loc[(dfBookClear['title'] == "Treasure Island")]


# We delete the duplicate lines seen above. 
# 11 123 - 10 891 = 232 (obtained above)

# ### Analysis of deleted duplicate data

# The fields used above in the `drop_duplicates()`: Title, Authors and language_code function have been defined since this data analysis. It checks if certain data is deleted and should not be. 
# For example, at the beginning of our analysis, we did not put the language_code column but thanks to this analysis, we could see that the function removed from books with the same title but not with the same language. In conclusion, these are interesting data to keep, we do not want to delete them. 

# In[20]:


dfDuplicatesRows.head(10)


# ## Removal of inconsistent data

# Using the `describe()` function, we noticed that some fields had data at 0 (min). In order to make the analysis more consistent, we decided to delete this data.

# In[21]:


dfBookClear.describe()


# In[22]:


dfBookClear.shape


# In[23]:


# Histogramm for average rating

fig = px.histogram(dfBookClear, x="average_rating", title="Average Rating", color_discrete_sequence =["#900C3F"])
fig.show()

# Histogram for numbers pages

fig = px.histogram(dfBookClear, x="num_pages", title="Numbers of pages", color_discrete_sequence =["#FF5733"])
fig.show()


# ##### Interpretation of the "Average Rating" histogram:

# We can interpret the average vote is between 3.5 and 4.5. We can say that the analysis will be based only between that range. For me, we don’t have a data set that is "broad enough" to have a precise analysis.  

# > Conclusion of the "Average Rating" histogram: A reader is likely to give a rating around 4.

# ##### Interpretation of the "Numbers Page" histogram:

# In addition, we chose to delete books with 0 pages so that our analysis is more accurate. That is inconsistent data.

# In[24]:


dfBookClear = dfBookClear.drop(dfBookClear[(dfBookClear['num_pages'] == 0)].index)


# In[25]:


dfBookClear.describe()


# In[26]:


dfBookClear.shape


# After removing books with 0 pages, we have 10817 books in our data set.

# # Prediction Model

# We try to predict the note of a book, for this we have the choice between two models of prediction:
#   - Classification model
#   - Regression model

# ## Classification Model

# The classification process searches for a function that helps divide the dataset into classes based on different parameters.

# In our case, we cannot have a classification process because the value we want to predict is not a class but a continuous value. 

# If our data set included the genres of books, we could have performed a clustering model to determine the genre of a book.

# ## Regression Model

# The regression process involves finding correlations between dependent and independent variables. It helps to predict continuous variables as in our case with the rating of a book. 

# Thanks to the `corr()` function, we can see which data is dependent or not. This is done only on the numerical columns (float, int, ...).

# In[27]:


dfBookClear.corr()


# In[28]:


fig = px.imshow(dfBookClear.corr(), aspect="auto")
fig.show()


# We can see through the heatmap the different correlations: 
#    - `ratings_count` and `text_reviews_count`: The number of notes depends on the number of comments. A reader will, most of the time, write down the book AND write a comment. We may think that adding a comment is mandatory to put a note.
#    - `pages` and `average_rating`: The average rating depends on the number of pages.

# #### Exploratory phase: Number of votes by number of pages

# In this section, we visualize the correlation between the number of pages of a book and its voting average.

# ##### Regression plot

# In[118]:


fig = px.scatter(dfBookClear, x="average_rating", y="num_pages", color='average_rating', title="Regression plot", trendline="ols")
fig.show()


# We can also use the `seaborn` library with the `regplot()` function to make our regression graph.

# In[119]:


plt.figure(figsize=(10,10))
sns.regplot(data=dfBookClear,y="average_rating",x="num_pages",marker='.',scatter_kws={"color": "#FFC300"}, line_kws={"color": "#581845"})
plt.xlabel('No. of Pages')
plt.ylabel('Average Rating')
plt.title("Regression plot")
plt.show()


# With the regression charts, we notice an upward trend in the rating relative to the number of pages. The reader tends to put higher notes on books with more pages.

# ##### Graphic to show outliers points.

# In[114]:


fig = px.scatter(dfBookClear, x="average_rating", y="num_pages", marginal_x="histogram", color_discrete_sequence = ['#F1C40F'], marginal_y="violin", title="Outliers")
fig.show()


# In[115]:


fig = px.box(dfBookClear, x="num_pages", color_discrete_sequence = ['#900C3F'], title="Outliers")
fig.show()


# All values above 1000 are off-centered points (outliers). Therefore, you must delete them to have a more accurate dataset.

# In[33]:


dfBookClear = dfBookClear.drop(dfBookClear.index[dfBookClear['num_pages'] >= 1000])


# In[116]:


fig = px.scatter(dfBookClear, x="average_rating", y="num_pages", marginal_x="histogram", color_discrete_sequence = ['#F1C40F'], marginal_y="violin", title="Outliers")
fig.show()


# ##### Results with regression charts

# We look with the regression charts to see if our approach above has impacted the trend between page count and voting average. We have smoothed our results by discarding out-of-center values.

# In[120]:


fig = px.scatter(dfBookClear, x="average_rating", y="num_pages", title="Regression plot", color='average_rating', trendline="ols")
fig.show()


# In[121]:


plt.figure(figsize=(10,10))
sns.regplot(data=dfBookClear,y="average_rating",x="num_pages",marker='.',scatter_kws={"color": "#FFC300"}, line_kws={"color": "#581845"})
plt.xlabel('No. of Pages')
plt.ylabel('Average Rating')
plt.title("Regression plot")
plt.show()


# To use our regression model, we can also correlate a specific column to the other columns: 

# In[37]:


dfBookClear.corr()["average_rating"]


# > Remark: The closer the result is to 1, the more the data is correlated with the data in the `average_rating` column

# If we want all columns to correlate, we need to change the `string` type columns to the `numeric` type, that is, standardize our dataset.

# ### Standardization 

# The standardization process is the conversion of data into a "standard" format.

# In[38]:


dfBookClear.info()


# We can see that the columns `title`, `authors`, `publication_date` are not numeric. 

# #### Transform the "string" fields into numbers:  Authors, Title , Publication_date and Language_code

# We will use the functions of the `sklearn` library to transform our data types. We make a copy of the basic data frame so as not to lose our previous data.

# In[39]:


from sklearn import preprocessing
dfBookEncoded = dfBookClear.copy() # copy of the dataframe dfBookClean for the create dataframe encoded
encoder = preprocessing.LabelEncoder()
dfBookEncoded['title'] = encoder.fit_transform(dfBookEncoded['title'])
dfBookEncoded['authors'] = encoder.fit_transform(dfBookEncoded['authors'])
dfBookEncoded['language_code'] = encoder.fit_transform(dfBookEncoded['language_code'])
dfBookEncoded['publication_date'] = encoder.fit_transform(dfBookEncoded['publication_date'])


# In[40]:


dfBookEncoded.sample(10)


# In[41]:


dfBookEncoded.sample(10)


# Now that we have implemented this process, we can look at the correlation only in the `average_rating` column with the other columns.

# In[42]:


dfBookEncoded.corr()["average_rating"].sort_values(ascending=False)


# In[43]:


dfBookEncoded.corr()


# In[44]:


fig = px.imshow(dfBookEncoded.corr(), aspect="auto")
fig.show()


# Despite the standardization of the Authors, Title, Publication_date and Language_code columns, we note that there are no additional correlations.

# #### Exploratory phase : Transform the "string" fields into numbers : Authors, Title and Publication_date

# We can try to define the `language_code` column as a category column to see if there is a better correlation. For this, we create another data frame without transforming the column.

# In[45]:


from sklearn import preprocessing
dfBookEncodedWithoutLanguageCode = dfBookClear.copy() # copy of the dataframe dfBookClean for the create dataframe encoded
encoder = preprocessing.LabelEncoder()
dfBookEncodedWithoutLanguageCode['title'] = encoder.fit_transform(dfBookEncoded['title'])
dfBookEncodedWithoutLanguageCode['authors'] = encoder.fit_transform(dfBookEncoded['authors'])
dfBookEncodedWithoutLanguageCode['publication_date'] = encoder.fit_transform(dfBookEncoded['publication_date'])


# The `get_dummies()` function is used to convert category variables to dummy variables. This can also be used to add column names with a prefix, for example. In our case, the category variable is <b> 'language_code'</b>:

# In[46]:


dfBookEncodedWithoutLanguageCode = pd.get_dummies(dfBookEncodedWithoutLanguageCode)


# In[47]:


dfBookEncodedWithoutLanguageCode.sample(10)


# In[48]:


dfBookEncodedWithoutLanguageCode.corr()["average_rating"].sort_values(ascending=False)


# In[49]:


fig = px.imshow(dfBookEncodedWithoutLanguageCode.corr(), aspect="auto")
fig.show()


# There is no relevant data from this `heatmap`. But we can deepen our analysis on the languages of books in an exploratory phase.

# ### Exploratory phase: Data analysis with the language_code column

# We can also analyze our data with the language of the books. We take back the basic data frame because in the two data frames we have just built, we no longer have languages in the form of strings but in the form of numbers.

# In[50]:


dfBookClear['language_code']


# In[51]:


dfBookClear['language_code'].isna().sum() #result 0
dfBookClear['language_code'].unique()


# With the `unique()` function, we can see the list of different languages assigned to books without duplicates.

# #### Highest-rated language and lower-rated languages

# We analyze the languages that have been rated the best and the least to see if the language of a book can be useful in our analysis.

# ##### The number of books per language

# Using the `value_counts()` function, we can see the number of books per language.

# In[52]:


dfBookClear['language_code'].value_counts()


# In[123]:


fig = px.pie(dfBookClear, values=dfBookClear['language_code'].value_counts(), title="Distribution of languages", names=dfBookClear['language_code'].unique(), color_discrete_sequence=px.colors.sequential.RdBu)
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# In[128]:


fig = px.histogram(dfBookClear,title="Distribution of languages", y="ratings_count", x="language_code")
fig.show()


# We can see the top 5 most used languages:

# In[55]:


dfBookClear['language_code'].value_counts().head()


# We find the values in percentages:

# In[56]:


dfCountValueLang = dfBookClear['language_code'].value_counts(normalize = True).head()
dfCountValueLang


# We notice that the most commonly used language is <b>'eng'</b>. And the least used languages are <b>'srp', 'nl', 'msa', 'glg', 'wel', 'ara', 'nor', 'tur', 'gla

# > Conclusion: A reader will be more likely to rate an English book.

# In[130]:


fig = px.scatter(dfBookClear, x="average_rating", y="num_pages", title="Average VS Num pages", color='language_code')
fig.show()


# Through the `scatter`, we see that the English language is everywhere on the graph, meaning that a lot more diverse votes have been allocated for that language. But this one is concentrated on the range of 3.5 and 4.5. 
# We clearly see that for any language, the voting range remains between 3.5 and 4.5. We find the same result as seen above in our analysis.  

# > Conclusion: We can predict that readers will likely score between 3.5 and 4.5.

# ## Model Training

# ### Choose predictors from columns

# Before we start making predictions, we need to select the relevant columns so that our prediction is as close as possible to reality. Based on our analysis, we only keep columns where `average_ratings` is correlated: `num_pages`, `text_review_counts`, and `ratings_count`. 

# In[58]:


dfTraining = dfBookClear.copy() # Copy of the data frame 
dfTraining = dfTraining.drop(columns=['title', 'authors', 'language_code', 'publication_date']) # delete unused columns 
target = "average_rating" # Value to predict 
dfTraining


# ### Split dataset : Train & Test

# In order to make our predictions, we need to separate our dataset into two: 
#    - "Train" is used to drive the model.
#    - "Test" is used to evaluate the model. 
#    
# To do this, we need to import the `train_test_split()` function from the `sklearn`:    

# In[61]:


from sklearn.model_selection import train_test_split
train = dfTraining.sample(frac=0.8, random_state=1) # Train dataset with 80% of the data.
test = dfTraining.loc[~dfTraining.index.isin(train.index)] # Test dataset with 20% of the data.


# In[62]:


train.shape


# In[63]:


test.shape


# We chose the Linear Regression and Random Forest models to perform the training and evaluation of the dataset. These regression models are known to perform well on unclassified data.

# ### Model Training : Linear Regression

# We are now trying to drive the linear regression model from the "train" dataset. For this we import the `LinearRegression()` class from the `sklearn` library: 

# In[64]:


from sklearn.linear_model import LinearRegression

modelReg = LinearRegression() # regression model

modelReg.fit(train[dfTraining.columns], train[target]) # model training


# ### Model Evaluation : Linear Regression

# After driving the model, we can calculate the prediction error rate using the `mean_squared_error` function in the `sklearn` library:

# In[65]:


predictionsReg = modelReg.predict(test[dfTraining.columns]) # prediction for the test data set.


# #### Mean Squared Error

# In[66]:


from sklearn.metrics import mean_squared_error
mean_squared_error(predictionsReg, test[target]) # calculate the error between predictions values and the reals values.


# Here we have an average accuracy of **2.64e-31**. This gives us a very low error rate. The closer the result is to 0, the better the predictions.

# #### Mean Absolute Error

# In[67]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(predictionsReg, test[target]) # calculate the error between predictions values and the reals values.


# The absolute average is **2.96e-16**. 

# #### Max Error

# In[68]:


from sklearn.metrics import max_error
max_error(predictionsReg, test[target])


# Our maximum error value is **3.12e-15**.

# #### Visualization of the prediction error

# To better represent the effectiveness of the prediction, we try to visualize it in a graphical form.

# In[131]:


dfVisual = pd.DataFrame({'Actual': test[target].tolist(), 'Prediction': predictionsReg.tolist()})
fig = px.scatter(dfVisual, x='Actual', y="Prediction", trendline='ols', title="Visualization of the prediction error", trendline_color_override="#900C3F")
fig.show()


# In this chart, we can see that all the points are on the trend line, which means that there are few error rates.

# ### Model Training : Random Forest

# We use the `RandomForestRegressor` function in the `sklearn` library:

# In[94]:


from sklearn.ensemble import RandomForestRegressor
# Initialiser le modèle avec certains paramètres.
modelForest = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
# Adapter le modèle aux données.
modelForest.fit(train[dfTraining.columns], train[target])


# ### Model Evaluation : Random Forest

# We generate Random Forest predictions from the test game.

# In[95]:


predictionsForest = modelForest.predict(test[dfTraining.columns])


# #### Mean Squared Error

# In[96]:


mean_squared_error(predictionsForest, test[target])


# We obtain a value <b>almost equal to 0</b> but always higher than the result of the linear regression model.

# #### Mean Absolute Error

# In[97]:


mean_absolute_error(predictionsForest, test[target])


# The absolute average is <b>0.002</b>. 

# #### Max Error

# In[98]:


max_error(predictionsForest, test[target])


# Our maximum error value is <b>0.61</b>.

# In[132]:


dfVisual = pd.DataFrame({'Actual': test[target].tolist(), 'Prediction': predictionsForest.tolist()})
fig = px.scatter(dfVisual, x='Actual', y="Prediction", trendline="ols", title="Visualization of the prediction error", trendline_color_override="#900C3F")
fig.show()


# In this graph, we observe that points are outside the trend line. 

# ## Conclusion

# We can conclude that the linear regression model is more reliable than the random forest model. There is more chance to predict a result close to the real, it will generally make less error. 
