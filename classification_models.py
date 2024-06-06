#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
import math

import operator
import pprint

from collections import defaultdict

class classifier_functions:
    def information_gain_target(dataset_file): 
        
    #        Input: dataset_file - A string variable which references the path to the dataset file.
    #        Output: ig_loan - A floating point variable which holds the information entropy associated with the target variable.
    #        
    #        NOTE: 
    #        1. Return the information gain associated with the target variable in the dataset.
    #        2. The Loan attribute is the target variable
    #        3. The pandas dataframe has the following attributes: Age, Income, Student, Credit Rating, Loan
    #        4. Perform your calculations for information gain and assign it to the variable ig_loan


        df = pd.read_csv(dataset_file)
        ig_loan = 0
        
        # your code here
        # Calculate the entropy of the target variable
        target = df['Loan']
        target_entropy = 0
        target_count = target.value_counts()
        # use ig formula to calculate for only the target variable
        for i in target_count:
            target_entropy += -i/len(target)*math.log2(i/len(target))
            
        ig_loan = target_entropy
        
        return ig_loan
    
    def information_gain(p_count_yes, p_count_no):
    #   A helper function that returns the information gain when given counts of number of yes and no values. 
    #   Please complete this function before you proceed to the information_gain_attributes function below.
        
        # your code here
        # calculate the information gain for 2 class variable with numbers of each class given as input
        ig = 0
        total = p_count_yes + p_count_no
        # check if there is no value in any of the classes
        if p_count_yes == 0 or p_count_no == 0:
            ig = 0
        else:
            # calculate ig using the formula
            ig = -((p_count_yes/total)*math.log2(p_count_yes/total) + (p_count_no/total)*math.log2(p_count_no/total))
            
        return ig

    def information_gain_attributes(dataset_file, ig_loan, attributes, attribute_values):
    #        Input: 
    #            1. dataset_file - A string variable which references the path to the dataset file.
    #            2. ig_loan - A floating point variable representing the information gain of the target variable "Loan".
    #            3. attributes - A python list which has all the attributes of the dataset
    #            4. attribute_values - A python dictionary representing the values each attribute can hold.
    #            
    #        Output: results - A python dictionary representing the information gain associated with each variable.
    #            1. ig_attributes - A sub dictionary representing the information gain for each attribute.
    #            2. best_attribute - Returns the attribute which has the highest information gain.
    #        
    #        NOTE: 
    #        1. The Loan attribute is the target variable
    #        2. The pandas dataframe has the following attributes: Age, Income, Student, Credit Rating, Loan
        
        results = {
            "ig_attributes": {
                "Age": 0,
                "Income": 0,
                "Student": 0,
                "Credit Rating": 0
            },
            "best_attribute": ""
        }
        
        df = pd.read_csv(dataset_file)
        d_range = len(df)
        
        for attribute in attributes:
            ig_attribute = 0
            value_counts = dict()
            vcount = df[attribute].value_counts()
            for att_value in attribute_values[attribute]:
                
                # your code here
                # get the count of each class in the attribute
                p_count_yes = 0
                p_count_no = 0
                # check if the value is in the attribute
                if att_value in vcount:
                    # get the count of the target variable for each class
                    p_count_yes = len(df[(df[attribute] == att_value) & (df['Loan'] == 'yes')])
                    p_count_no = len(df[(df[attribute] == att_value) & (df['Loan'] == 'no')])
                # calculate the information gain for the attribute
                ig_attribute += (vcount[att_value]/d_range)*information_gain(p_count_yes, p_count_no)
                
            
            results["ig_attributes"][attribute] = ig_loan - ig_attribute
            
        
        results["best_attribute"] = max(results["ig_attributes"].items(), key=operator.itemgetter(1))[0]
        return results
    
    from collections import defaultdict

    def naive_bayes(dataset_file, attributes, attribute_values):
    #   Input:
    #       1. dataset_file - A string variable which references the path to the dataset file.
    #       2. attributes - A python list which has all the attributes of the dataset
    #       3. attribute_values - A python dictionary representing the values each attribute can hold.
    #        
    #   Output: A probabilities dictionary which contains the values of when the input attribute is yes or no
    #       depending on the corresponding Loan attribute.
    #                
    #   Hint: Starter code has been provided to you to calculate the probabilities.

        probabilities = {
            "Age": { "<=30": {"yes": 0, "no": 0}, "31-40": {"yes": 0, "no": 0}, ">40": {"yes": 0, "no": 0} },
            "Income": { "low": {"yes": 0, "no": 0}, "medium": {"yes": 0, "no": 0}, "high": {"yes": 0, "no": 0}},
            "Student": { "yes": {"yes": 0, "no": 0}, "no": {"yes": 0, "no": 0} },
            "Credit Rating": { "fair": {"yes": 0, "no": 0}, "excellent": {"yes": 0, "no": 0} },
            "Loan": {"yes": 0, "no": 0}
        }
        
        df = pd.read_csv(dataset_file)
        d_range = len(df)
        
        vcount = df["Loan"].value_counts()
        vcount_loan_yes = vcount["yes"]
        vcount_loan_no = vcount["no"]
        
        probabilities["Loan"]["yes"] = vcount_loan_yes/d_range
        probabilities["Loan"]["no"] = vcount_loan_no/d_range
        
        for attribute in attributes:
            value_counts = dict()
            vcount = df[attribute].value_counts()
            for att_value in attribute_values[attribute]:
                
                # your code here
                if att_value in vcount:
                    # get the count by target variable for each class
                    vcount_attr_yes = len(df[(df[attribute] == att_value) & (df['Loan'] == 'yes')])
                    vcount_attr_no = len(df[(df[attribute] == att_value) & (df['Loan'] == 'no')])
                    # get expected probability for each class
                    prob_target_yes = probabilities["Loan"]["yes"]
                    prob_target_no = probabilities["Loan"]["no"]
                    # posterior probability of the attribute given the target variable
                    prob_target_given_attr_yes = vcount_attr_yes/vcount_loan_yes
                    prob_target_given_attr_no = vcount_attr_no/vcount_loan_no
                    
                    probabilities[attribute][att_value]["yes"] = prob_target_given_attr_yes
                    probabilities[attribute][att_value]["no"] = prob_target_given_attr_no
        
        return probabilities
    


if __name__ == "__main__":
    cf = classifier_functions()
    
    ig_loan = cf.information_gain_target('./data/dataset.csv')
    ig_loan_expected = 0.9798687566511528

    print(f'The expected ig_loan value for the given dataset is: {ig_loan_expected}')
    print(f'Your ig_loan value is: {ig_loan}')

    try:
        np.testing.assert_allclose(ig_loan, ig_loan_expected, rtol=0.001, atol=0.001)
        print("Visible tests passed!")
    except:
        print("Visible tests failed!")
        
    attribute_values = {
    "Age": ["<=30", "31-40", ">40"],
    "Income": ["low", "medium", "high"],
    "Student": ["yes", "no"],
    "Credit Rating": ["fair", "excellent"]
    }

    attributes = ["Age", "Income", "Student", "Credit Rating"]
    
    # This cell has visible test cases that you can run to see if you are on the right track!
    # Note: hidden tests will also be applied on other datasets for final grading.

    import pprint
    pp = pprint.PrettyPrinter(depth=4)
    ig_loan_expected = 0.9798687566511528

    attribute_values = {
        "Age": ["<=30", "31-40", ">40"],
        "Income": ["low", "medium", "high"],
        "Student": ["yes", "no"],
        "Credit Rating": ["fair", "excellent"]
    }

    attributes = ["Age", "Income", "Student", "Credit Rating"]

    results = cf.information_gain_attributes("./data/dataset.csv", ig_loan_expected, attributes, attribute_values)

    results_expected = {'ig_attributes': {'Age': 0.2419726756283742, 'Income': 0.012398717114751934, 'Student': 0.19570962879973097, 'Credit Rating': 0.07181901063117269}, 'best_attribute': 'Age'}

    print(f'The expected results value for the given dataset is:')
    pp.pprint(results_expected)
    print(f'Your results value is:')
    pp.pprint(results)

    try:
        x = pd.Series(results["ig_attributes"])
        y = pd.Series(results_expected["ig_attributes"])
        pd.testing.assert_series_equal(x, y, check_exact = False)  #check_less_precise=3
        assert results["best_attribute"] == results_expected["best_attribute"]
        print("Visible tests passed!")
    except AssertionError as e:
        # if this was a shape or index/col error, then re-raise
        try:
            pd.testing.assert_index_equal(x.index, y.index)
            pd.testing.assert_index_equal(x.columns, y.columns)
        except AssertionError:
            raise e

        # if not, we have a value error 
        diff = x != y
        diffcols = diff.any(axis=0)
        diffrows = diff.any(axis=1)
        cmp = pd.concat(
            {'left': x.loc[diffrows, diffcols], 'right': y.loc[diffrows, diffcols]},
            names=['dataframe'],
            axis=1,
        )

        raise AssertionError(e.args[0] + f'\n\nDifferences:\n{cmp}') from None
    # except:
    #     print("Visible tests failed!")

    pp = pprint.PrettyPrinter(depth=6)

    attribute_values = {
        "Age": ["<=30", "31-40", ">40"],
        "Income": ["low", "medium", "high"],
        "Student": ["yes", "no"],
        "Credit Rating": ["fair", "excellent"]
    }

    attributes = ["Age", "Income", "Student", "Credit Rating"]

    probabilities = cf.naive_bayes("./data/dataset.csv", attributes, attribute_values)

    probabilities_expected = {'Age': {'<=30': {'yes': 0.2857142857142857, 'no': 0.6},
    '31-40': {'yes': 0.42857142857142855, 'no': 0.0},
    '>40': {'yes': 0.2857142857142857, 'no': 0.4}},
    'Income': {'low': {'yes': 0.2857142857142857, 'no': 0.2},
    'medium': {'yes': 0.42857142857142855, 'no': 0.4},
    'high': {'yes': 0.2857142857142857, 'no': 0.4}},
    'Student': {'yes': {'yes': 0.7142857142857143, 'no': 0.2},
    'no': {'yes': 0.2857142857142857, 'no': 0.8}},
    'Credit Rating': {'fair': {'yes': 0.7142857142857143, 'no': 0.4},
    'excellent': {'yes': 0.2857142857142857, 'no': 0.6}},
    'Loan': {'yes': 0.5833333333333334, 'no': 0.4166666666666667}}

    print(f'Your probabilities value is:')
    pp.pprint(probabilities)
    print(f'\nThe expected probabilities value for the given dataset is:')
    pp.pprint(probabilities_expected)

    try:
        for i in attributes:
            for j in attribute_values[i]:
                for k in ["yes", "no"]:
                    np.testing.assert_allclose(probabilities[i][j][k], probabilities_expected[i][j][k], rtol=0.001, atol=0.001)
        print("Visible tests passed!")
    except:
        print("Visible tests failed!")