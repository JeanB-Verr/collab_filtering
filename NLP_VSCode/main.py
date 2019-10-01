# Library containing methods to perform collaberative filtering
import surprise
from surprise import KNNWithMeans
from surprise import Dataset
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate, train_test_split
from surprise import Reader, Dataset, SVD
from surprise import accuracy
# Needed to create new dataset, where ratings are random values
import random
from random import randint
# Strip content from text
import re
# Panda for data manipulation
import pandas as pd
import timeit
import os

class Recommendation(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        with open(self.dataset_path) as f:
            read_file = f.readlines()

        self.read_file = read_file

        # Prepare the data to be used in Surprise
        reader = Reader(line_format='user item rating timestamp', sep='\t')
        data = Dataset.load_from_file(dataset_path, reader=reader)
        self.data = data

    # Perform grid search in order to find the best parameters resulting in the lowest RMSE.
    def grid_search(self):
        param_grid = {
            "n_epochs": [5, 10],
            "lr_all": [0.001, 0.002, 0.005],
            "reg_all": [0.1, 0.2, 0.4, 0.6]
        }
        gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)

        gs.fit(self.data)

        print(gs.best_score["rmse"])
        print(gs.best_params["rmse"])

    def create_algo(self, epochs, lr, reg):
        algo = SVD(n_epochs = epochs, lr_all = lr, reg_all = reg)
        self.algo = algo

    def cross_validation(self, n_folds):
        cross_validate(self.aglo, self.data, measures=['RMSE', 'MAE'], cv=n_folds, verbose=True)

    def RMSE_predict_train_test(self, size_test):
        # Test set is made of 25% of the ratings.
        trainset, testset = train_test_split(self.data, test_size=.25)

        # Train the algorithm on the trainset, and predict ratings for the testset
        algo.fit(trainset)
        predictions = self.algo.test(testset)

        return accuracy.rmse(predictions)

    def train_algo(self):
        start = timeit.default_timer()

        trainset = self.data.build_full_trainset()
        self.trainset = trainset

        # Get all the user and item IDs
        user_ids = self.trainset.all_users()
        item_ids = self.trainset.all_items()
        self.user_ids = user_ids
        self.item_ids = item_ids

        self.algo.fit(self.trainset)

        stop = timeit.default_timer()
        print('Time: ', stop - start) 


    def predict_scores(self):
        start = timeit.default_timer()

        # Create empty list to store predictions
        ratings = []
        
        # For loop, estimate rating of each user for every movie.
        for user_id in self.user_ids:
            for item_id in self.item_ids:
                prediction =self.algo.predict(str(user_id), str(item_id)).est
                ratings.append(prediction)
                
        stop = timeit.default_timer()
        print('Time: ', stop - start)  
        
        return ratings

    # The data contains line breaks. Remove the line breaks in order to create clean text.
    def strip_content(self):
        r_unwanted = re.compile("[\n\t\r]")
        return r_unwanted.sub(", ", self.data)

    def create_dataframe(self):
        # Create DataFrame
        df_data = pd.DataFrame(data)

        # Apply strip content function on every row of the DataFrame
        df_data[0] = df_data[0].apply(strip_content)

        # Generate comma seperation for column generation
        foo = lambda x: pd.Series([i for i in (x.split(','))])
        df_final = df_data[0].apply(foo)

        df_final.rename(columns={0:'userID', 1:'movieID', 2:'rating', 3: 'timestamp'}, 
                        inplace=True)
        df_final = df_final.drop(columns=[4])
        
        return df_final
        
    def predict_scores_df(self):
        start = timeit.default_timer()
   
        # Create empty list to store predictions
        ratings = {}
        ratings_list = []
        
        # For loop, estimate rating of each user for every movie.
        for user_id in self.user_ids:
            for item_id in self.item_ids:
                
                prediction = self.algo.predict(str(user_id), str(item_id)).est
                ratings['userID'] = str(user_id)
                ratings['movieID'] = str(item_id)
                ratings['rating'] = prediction
                
                ratings_list.append(ratings)
                
                ratings = {}

        stop = timeit.default_timer()
        print('Time: ', stop - start)  
        return ratings_list

    def create_data(self, users_per_movie):
        data_list = []
        # Create new movies (168200 in total)
        for movie in range(self.item_ids[-1]+1, self.item_ids[-1]*users_per_movie):
            # For every movie, there will be 100 users rating the movie
            user_generated = [randint(0, self.user_ids[-1]) for p in range(0, 100)]
            for user in user_generated:
                # Create a random generated score for the movies
                new_data = str(user)+'\t'+str(movie)+'\t'+str(random.randint(1,5))+'\t'+'NaN\n'

                data_list.append(new_data)

        new_data = []
        new_data.extend(self.read_file)
        new_data.extend(data_list)

        return new_data
    
    def save_data(self, file_name, new_data):
        with open('{}.data'.format(file_name), 'w') as f:
            f.write(new_data)
    
    def load_data(self):
        pass
    

if __name__ == "__main__":
    path = "C://Users//Jean-Baptiste.Verrok//Documents//NLP//ml-100k//u.data"
    collab_filter = Recommendation(path)
    collab_filter.create_algo(epochs = 10, lr = 0.0056, reg = 0.4)
    collab_filter.train_algo()
    scores = collab_filter.predict_scores()
    print(scores[0])



