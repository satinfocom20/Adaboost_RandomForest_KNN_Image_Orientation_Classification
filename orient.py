import sys
import csv
import math
import random
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import numpy as np
import pickle
import time

class ml:
    def __init__(self):
        self.vector = []
        self.orientation = []
        self.id = []
        self.weights = []

    def file_read(self, file_name):
        f = open(file_name)
        reader = csv.reader(f, delimiter=" ")
        for line in reader:
            self.id.append(line[0])
            self.orientation.append(int(line[1]))
            self.vector.append(np.asarray(list(map(int, line[2:]))))

    # ------------Start of K Nearest neighbour-----------------------------------------

    #Train the model by just loading the image pixel vector and its orientations into model file
    def train_nn(self, train_file, model_file):
        self.file_read(train_file)
        data = {'vectors': np.asarray(self.vector), 'orientation': self.orientation}
        np.save(model_file, data)

    #Select a neighbour with maximum vote to classify that as a final predicted orientation
    def getMaxVote(self, neighbours):
        nneighbour_count = {}
        for i in neighbours:
            try:
                nneighbour_count[i[1]] += 1
            except:
                nneighbour_count[i[1]] = 1
        return max(zip(nneighbour_count.keys(), nneighbour_count.values()))

    #Test nearest neighbour with test dataset
    def test_nn(self, test_file, model_file, k=1):
        #Load the model file
        train_data = np.load(model_file)
        train_data = train_data[()]
        #Retrieve the image pixel vectors and orientation into a list
        train_vec = train_data['vectors']
        train_class = train_data['orientation']
        #Read test records and load id, image pixel vector and orientation into separate lists
        self.file_read(test_file)
        #Convert the image pixel vector from from list to numpy array for faster computation
        self.vector = np.asarray(self.vector)
        output_nearest = []
        correct = 0
        #Calculate eulidean distance for the each test image pixel vector against all train vectors
        #to find 'k=1' vectors with shortest distance and take majority voting among the shorlisted
        #vectors and pick the orientation of the selected train vector to classify test vector
        for c, vector in enumerate(self.vector):
            dist = self.getEucledianDist(train_vec, vector)
            neighbours = sorted(zip(dist, train_class), key=lambda tup: tup[0])
            neighbours = neighbours[:k]
            max_class_count = self.getMaxVote(neighbours)[0]
            output_nearest.append(str(self.id[c]) + " " + str(max_class_count))
            if max_class_count == self.orientation[c]:
                correct += 1
        print ("Nearest neighbour accuracy on test dataset: ", (correct / float(len(self.vector))) * 100, "%")
        with open('nearest_output.txt', 'wt') as nearest_file:
            nearest_file.write("\n".join(output_nearest))

    #Calculate eucledian distance
    def getEucledianDist(self, vec1, vec2):
        return np.sum(abs(vec2 - vec1), axis=1)


    #------------Start of Adaboost-----------------------------------------
    #Train the test dataset
    def train_boost(self, train_file, model_file):
        #Separate photo id, orientation and vector of pixels from the train file
        self.file_read(train_file)
        #Initialize weights
        self.weights = [1/float(len(self.vector))]*len(self.vector)
        #Call hypothesis 1 and 2 to find which has maximum accuracy
        weights_hypo = {}
        for hypo_no in range(1,3):
            hypo_selected, hypo_prediction = self.eval_hypo(hypo_no)
            #Calculate new weights and alpha(overall weight) for hypothesis
            error = 0
            for i in range(len(self.vector)):
                #add the weights of mis classified exemplars
                if hypo_prediction[hypo_selected][i] != self.orientation[i]:
                    error += self.weights[i]
            total_error = error

            for i in range(len(self.vector)):
                #Increase the weights of correctly classified exemplars
                if hypo_prediction[hypo_selected] == self.orientation[i]:
                    self.weights[i] = self.weights[i] * (total_error/(1 - total_error))
            #Normalize the updated weights
            self.weights = [float(w)/sum(self.weights) for w in self.weights]
            #Calculate final weight of hypothesis
            weights_hypo['hypo'+str(hypo_no)+'-weight'] = math.log((1 - total_error) / total_error)

        #Write the hypothesis weights in the model file
        filew = open(model_file, 'a')
        for k, v in weights_hypo.items():
            filew.write('%s %.9f\n' % (k, v+10))

    #Evaluate hypothesis 1 and 2 to find the one with max accuracy
    def eval_hypo(self, hypo_no):
        pred_hypo = {}
        hypo_accuracy = {'hypo' + str(i): 0 for i in range(1, 3)}
        for hypo in range(hypo_no,3):
            correct = 0
            pred = self.hypothesis(hypo)
            pred_hypo['hypo'+str(hypo)] = pred
            for i in range(len(self.orientation)):
                if pred[i] == self.orientation[i]:
                    correct += self.weights[i]
            hypo_accuracy['hypo'+str(hypo)] = correct
        return max(hypo_accuracy, key=hypo_accuracy.get), pred_hypo

    # Hypothesis 1 - Find which position of image has maximum blue pixel concentration
    def hypothesis(self, hypo):
        pred_orient = []

        if hypo == 1:
            for vec in self.vector:
                top, right, bottom, left = sum(vec[2:24:3]), sum(vec[23:192:24]), \
                                           sum(vec[170:192:3]), sum(vec[2:192:24])
                if max([top, right, bottom, left]) == top:
                    pred_orient.append(0)
                elif max([top, right, bottom, left]) == bottom:
                    pred_orient.append(180)
                elif max([top, right, bottom, left]) == right:
                    pred_orient.append(90)
                elif max([top, right, bottom, left]) == left:
                    pred_orient.append(270)
            #return pred_orient
        elif hypo == 2:
        # Hypothesis 1 - Find which position of image has maximum brown pixel concentration
            for vec in self.vector:
                top, right, bottom, left = sum(vec[0:23:3]), sum(vec[21:192:24]), \
                                           sum(vec[168:192:3]), sum(vec[0:192:24])
                if max([top, right, bottom, left]) == top:
                    pred_orient.append(180)
                elif max([top, right, bottom, left]) == bottom:
                    pred_orient.append(0)
                elif max([top, right, bottom, left]) == right:
                    pred_orient.append(270)
                elif max([top, right, bottom, left]) == left:
                    pred_orient.append(90)
        else:
            print('Invalid hypothesis option')

        return pred_orient

    def test_boost(self, test_file, model_file, model):
        # Separate photo id, orientation and vector of pixels from the train file
        self.file_read(test_file)
        #Read the model file for the hypothesis weights
        filer = open(model_file, 'r')
        model_lines = filer.readlines()
        hypo_wt = {}
        final_predictions = {}
        hypo_predictions = {}
        for line in model_lines:
            line = line.strip().split(' ')
            hypo_wt[str(line[0])] = float(line[1])

        #Run the test file image vectors in both hypothesis and capture the predicted orientation
        hypo_predictions = {i: [] for i in range(len(self.vector))}
        for hypo in range(1,3):
            pred = self.hypothesis(hypo)
            for idx, orient in enumerate(pred):
                hypo_predictions[idx].append((orient, hypo_wt['hypo'+str(hypo)+'-weight']))

        #For each test file image add weights for each predicted orientation from both hpothesis and take
        #select the one with max weight for the final prediction
        for tstrec, hypoval in hypo_predictions.items():
            vote_orient = {}
            for orient, hypowgt in hypoval:
                try:
                    vote_orient[orient] += hypowgt
                except:
                    vote_orient[orient] = hypowgt
            final_predictions[tstrec] = max(vote_orient, key=vote_orient.get)

        # print accuracy of the prediction
        correct = 0
        for i in range(len(self.orientation)):
            if self.orientation[i] == final_predictions[i]:
                correct += 1
        output_file = 'adaboost_output.txt'
        if model == 'adaboost':
            print("Adaboost accuracy on test set: ", (correct / float(len(self.orientation))) * 100, "%")
            output_file = 'adaboost_output.txt'
        elif model == 'best':
            print("Best alogorithm Adaboost's accuracy on test set: ", (correct / float(len(self.orientation))) * 100, "%")
            output_file = 'best_output.txt'
        #Write the final predicted orientation into the output file
        filew = open(output_file, 'a')
        for idx, imageid in enumerate(self.id):
            filew.write('%s %d\n' %(imageid, final_predictions[idx]))

        return final_predictions


    # ------------Start of Random forest -----------------------------------------
    # Load a CSV file
    def load_csv(self, filename):
        dataset = list()
        with open(filename, 'r') as file:
            csv_reader = reader(file, delimiter=' ')
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset

    # Convert string column to int
    def str_feature_to_int1(self, dataset, column):
        for row in dataset:
            row[column] = int(row[column].strip())

    # Convert string column to integer
    def str_feature_to_int(self, dataset, column):
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
        for row in dataset:
            row[column] = lookup[row[column]]
        return lookup

    # Calculate accuracy percentage
    def RF_Acurracy(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0


    # Evaluate an algorithm using a cross validation split
    def RF_Evaluation(self, dataset, *args):
        test_set = []
        train_set = dataset
        trees = self.RandomForest(train_set, test_set, *args)
        return trees

    # Split a dataset based on an attribute and an attribute value
    def RF_Split(self, index, value, dataset):
        # def RF_Split(index, value, row):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                # if row[index] < 127:
                left.append(row)
            else:
                right.append(row)
        '''if value < 127:
            left.append(row)
        else:
            right.append(row)'''

        return left, right

    # Calculate the Gini index for a split dataset
    def RF_Gini(self, groups, classes):
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    # Select the best split point for a dataset
    def RF_Best_Split(self, dataset, nfeatures):
        class_values = np.unique(np.transpose(dataset)[0])
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        features = list()
        while len(features) < nfeatures:
            index = randrange(2, (len(dataset[0]) - 1))
            if index not in features:
                features.append(index)
        for index in features:
            value = random.choice([64,127, 190])
            # t1 = time.time()
            groups = self.RF_Split(index, value, dataset)
            # print('test split time ', time.time() - t1)
            gini = self.RF_Gini(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, value, gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    # Create a terminal node value
    def terminal_node(self, group):
        outcomes = [row[0] for row in group]
        return max(set(outcomes), key=outcomes.count)

    # Create child splits for a node or make terminal
    def split(self, node, max_depth, min_size, nfeatures, depth):
        left, right = node['groups']
        del (node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.terminal_node(left + right)
            return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.terminal_node(left), self.terminal_node(right)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = self.terminal_node(left)
        else:
            node['left'] = self.RF_Best_Split(left, nfeatures)
            self.split(node['left'], max_depth, min_size, nfeatures, depth + 1)
        # process right child
        if len(right) <= min_size:
            node['right'] = self.terminal_node(right)
        else:
            node['right'] = self.RF_Best_Split(right, nfeatures)
            self.split(node['right'], max_depth, min_size, nfeatures, depth + 1)

    # Build a decision tree
    def RF_Buildtree(self, train, max_depth, min_size, nfeatures):
        root = self.RF_Best_Split(train, nfeatures)
        t2 = time.time()
        self.split(root, max_depth, min_size, nfeatures, 1)
        print('split time ', time.time() - t2)
        return root

    # Make a prediction with a decision tree
    def DT_predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.DT_predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.DT_predict(node['right'], row)
            else:
                return node['right']

    # Create a random subsample from the dataset with replacement
    def RF_subsample(self, dataset, ratio):
        sample = list()
        n_sample = round(len(dataset) * ratio)
        while len(sample) < n_sample:
            index = randrange(len(dataset))
            d = dataset[index]
            d = np.asarray(list(map(int, d[1:])))
            # sample.append(dataset[index])
            sample.append(d)
        return sample

    # Make a prediction with a list of bagged trees
    def RF_predict(self, trees, row):
        predictions = [self.DT_predict(tree, row) for tree in trees]
        return max(set(predictions), key=predictions.count)

    # Random Forest Algorithm
    def RandomForest(self, train, test, max_depth, min_size, sample_size, ntrees, nfeatures):
        trees = list()
        for i in range(ntrees):
            sample = self.RF_subsample(train, sample_size)
            tree = self.RF_Buildtree(sample, max_depth, min_size, nfeatures)
            #print('Built Tree ', i)
            trees.append(tree)
        #print('Trees: ', trees)
        return trees

    #Train random forest algorithm
    def train_RandomForest(self, train_file, model_file):
        # Test the random forest algorithm
        seed(2)
        # load and prepare data
        # train_filename = 'train-data.txt'
        dataset = self.load_csv(train_file)
        # convert string attributes to integers
        for i in range(1, len(dataset[0])):
            self.str_feature_to_int1(dataset, i)
        # evaluate algorithm
        n_folds = 2
        max_depth = 12
        min_size = 1
        sample_size = 1.0
        nfeatures = int(sqrt(len(dataset[0]) - 1))
        ntrees = 300
        # scores, trees = RF_Evaluation(dataset, RandomForest, n_folds, max_depth, min_size, sample_size, ntrees, nfeatures)
        trees = self.RF_Evaluation(dataset, max_depth, min_size, sample_size, ntrees, nfeatures)
        with open(model_file, "wb") as fp:  # Pickling
            pickle.dump(trees, fp, protocol=2)
        print('Trees: %d' % ntrees)

    #Test run of random forest algorithm
    def test_RandomForest(self, test_file, model_file):
        tree_model = []
        test = m.load_csv(test_file)
        with open(model_file, "rb") as fp:  # Unpickling
            tree_model = pickle.load(fp)
        # convert string attributes to integers
        for i in range(1, len(test[0])):
            m.str_feature_to_int1(test, i)
        test_predictions = [m.RF_predict(tree_model, row) for row in test]
        #print('Test predictions \n', test_predictions)
        actual = [row[1] for row in test]
        accuracy = m.RF_Acurracy(actual, test_predictions)
        print('Random forest Test Accuracy: ', accuracy)

        # Write the final predicted orientation into the output file
        filew = open('forest_output.txt', 'a')
        for idx, row in enumerate(test):
            filew.write('%s %d\n' % (row[0], test_predictions[idx]))


#---------Implementation-------------------------
m = ml()

#----Run using IDE
#--Call K nearest neighbour
#m.train_nn('train-data.txt', 'model-file')
#m.test_nn('test-data.txt', 'model-file.npy')

#--Call Adaboost
#m.train_boost('train-data.txt', 'model-boost-file.txt')
#m.test_boost('test-data.txt', 'model-boost-file.txt')

#--Call Random forest
#m.train_boost('traindata.txt', 'model-boost-file.txt')
#m.test_boost('test-data.txt', 'model-boost-file.txt')
#phase = 'train'
phase = 'test'
#train_file = 'train-data.txt'
test_file = 'test-data.txt'
model_file = 'forest_model.txt'
model = 'forest'

#-----------Run using command line
'''phase = sys.argv[1]
if phase == 'train':
    train_file = sys.argv[2]
elif phase == 'test':
    test_file = sys.argv[2]
else:
    print('Invalid phase option')
model_file = sys.argv[3]
model = sys.argv[4] '''

if model == 'nearest':
    if phase == 'train':
        m.train_nn(train_file, model_file)
    elif phase == 'test':
        model_file = model_file+'.npy'
        m.test_nn(test_file, model_file)
    else:
        print('Invalid phase option')
elif model == 'adaboost' or model == 'best':
    if phase == 'train':
        m.train_boost(train_file, model_file)
    elif phase == 'test':
        m.test_boost(test_file, model_file, model)
    else:
        print('Invalid phase option')
elif model == 'forest':
    if phase == 'train':
        m.train_RandomForest(train_file, model_file)
    elif phase == 'test':
        m.test_RandomForest(test_file, model_file)
    else:
        print('Invalid phase option')
else:
    print('Invalid model option')


