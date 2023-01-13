import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score, auc

import numpy as np
import math

# Data Mining
class Model:
    def __init__(self, models = None):
        self.models = {} if not models else models
        
    def add(self, model_name, model):
        self.models[model_name] = model
        
class Metric:
    def __init__(self, X_train, X_test, y_train, y_test, class_type = "multi"):
        self.X_train = X_train
        self.X_test = X_test
        
        self.y_train = y_train
        self.y_test = y_test
        
        self.average = "macro" if class_type == "multi" else "binary"
    
        self.scores = {}
        self.cm = {}
        
    def multi_auc(self, model_name, prob):
        classes = self.y_test.unique().tolist()
        
        for p in classes:
            fpr, tpr, _ = roc_curve(self.y_test, prob, pos_label = p) 

            auc_score = auc(fpr, tpr)
            self.scores[model_name].append(auc_score)
            self.index.append(f"AUC Score of Class {p}") 

    def score(self, model_name = None, model = None, how = "all", models = None, X_train = None, X_test = None, y_train = None, y_test = None):
        
        if X_train is None: X_train = self.X_train
            
        if X_test is None: X_test = self.X_test
            
        if y_train is None: y_train = self.y_train
            
        if y_test is None: y_test = self.y_test

        
        if model is not None: 
            models = {model_name : model}
            
        elif how == "all": models = models.models
        for model_name, model in models.items():
           
            
            model.fit(X_train, y_train)
            
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
                
            y_pred = model.predict(X_test)
            
            precision = precision_score(y_test, y_pred, average = self.average)
            recall = recall_score(y_test, y_pred, average = self.average)
            f1 = f1_score(y_test, y_pred, average = self.average)
            
            try: 
                prob = model.decision_function(X_test)
                
            except:
                prob = model.predict_proba(X_test)[:, 1]
                
             
            self.scores[model_name] = [train_score, test_score, precision, recall, f1]
       
            self.cm[model_name] = y_pred
            
            self.index = ["Training Accuracy", "Testing Accuracy", "Precision", "Recall", "F1 Score"]
            
            if self.average == "macro": self.multi_auc(model_name, prob)
            else: 
                auc = roc_auc_score(y_test, prob)
           
                
                self.scores[model_name].append(auc)
                self.index.append("AUC Score")
              
        res = pd.DataFrame(self.scores, index = self.index).T
                                
        return res
    
    
    def compute_fig_rows(self, n_models, n_cols):
        n_rows = math.ceil(n_models / n_cols)
        if n_rows == 0: n_rows = 1
        
        return n_rows
    
    def conf_mat(self, figsize = (15, 6), cmap = "Blues"):
        label_names = self.y_test.unique()
        shape = len(label_names)
        
        n_models = len(self.cm)
        n_rows = self.compute_fig_rows(n_models, 2)
        n_cols = 2
            
        fig, axes = plt.subplots(n_rows, n_cols, figsize = figsize)
        
        for ax, model_name, y_pred in zip(axes.flatten(), self.cm.keys(), self.cm.values()):
            cf_matrix = confusion_matrix(self.y_test, y_pred)
            group_counts = [val for val in cf_matrix.flatten()]
            group_percent = ["{:.2%}".format(val) for val in cf_matrix.flatten() / np.sum(cf_matrix)]

            labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percent)]
            labels = np.asarray(labels).reshape(shape, shape)
            
            ax.set_title(model_name)
            sns.heatmap(cf_matrix, annot = labels, cmap = cmap, fmt = "", xticklabels = label_names,yticklabels = label_names, ax = ax)
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")
        
        if n_models % 2 != 0:
            if n_models < 3: axes[-1].set_visible(False)
            else: axes[-1][-1].set_visible(False)
            
        fig.tight_layout()
                                                                                                                  