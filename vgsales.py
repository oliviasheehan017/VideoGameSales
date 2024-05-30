import pandas as pd #importing pandas
from sklearn.tree import DecisionTreeClassifier, export_graphviz #this is to visualize the decision tree
from sklearn.model_selection import train_test_split #helps us train the date
from sklearn.metrics import accuracy_score
import joblib

vg_data = pd.read_csv('vgsales.csv') #reading csv file of the data


vg_data_subset = vg_data.head(100).copy() #cutting my data down to only the first 100 

vg_data_subset.dropna(subset=['Publisher'], inplace=True) #drop publisher from the input table so we can use it for output

X = vg_data_subset.drop(columns=['Name', 'Platform', 'Year', 'Genre', 'Publisher','Other_Sales']) #dropping unnecessary columns

y = vg_data_subset['Publisher'] #this will be the output


# pred = model.predict([ [88, 4.0, 2.7, 0.4, 6.2],[2, 40.8 ,14.2, 8.7, 30.3]]) #giving the model some made up data to predict the publisher name
# print('predictions for random input data:')
# print(pred) #print

#print(y_test, predictions) #generate the accuracy score

print('This model will predict the video game publisher based on video game sales and rank')
print('accuracy score is:')
score = accuracy_score(y_test, predictions)
score


# model = joblib.load('vg_publisher_predictions.joblib') #load the model from the joblib if we want to use it in the future

#export_graphviz(model, out_file='vg_publisher_predictions.dot', feature_names= ['Rank', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Global_Sales'], class_names=sorted(y.unique()),label='all', rounded=True, filled=True) #making it a visual tree
