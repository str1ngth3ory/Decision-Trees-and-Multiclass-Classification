import numpy as np
from collections import Counter

class DecisionNode():
    """Class to represent a single node in
    a decision tree."""

    def __init__(self, left, right, decision_function,class_label=None):
        """Create a node with a left child, right child,
        decision function and optional class label
        for leaf nodes."""
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Return on a label if node is leaf,
        or pass the decision down to the node's
        left/right child (depending on decision
        function)."""
        if self.class_label is not None:
            return self.class_label
        elif self.decision_function(feature):
            return self.left.decide(feature)
        else:
            return self.right.decide(feature)

def build_decision_tree():
    """Create decision tree
    capable of handling the provided 
    data."""
    # TODO: build full tree from root
    decision_tree_root = None
    
    return decision_tree_root

def confusion_matrix(classifier_output, true_labels):
    #TODO output should be [[true_positive, false_negative], [false_positive, true_negative]]
    raise NotImplemented()

def precision(classifier_output, true_labels):
    #TODO precision is measured as: true_positive/ (true_positive + false_positive)
    raise NotImplemented()
    
def recall(classifier_output, true_labels):
    #TODO: recall is measured as: true_positive/ (true_positive + false_negative)
    raise NotImplemented()
    
def accuracy(classifier_output, true_labels):
    #TODO accuracy is measured as:  correct_classifications / total_number_examples
    raise NotImplemented()

def entropy(class_vector):
    """Compute the entropy for a list
    of classes (given as either 0 or 1)."""
    # TODO: finish this
    raise NotImplemented()
    
def information_gain(previous_classes, current_classes ):
    """Compute the information gain between the
    previous and current classes (each 
    a list of 0 and 1 values)."""
    # TODO: finish this
    raise NotImplemented()

class DecisionTree():
    """Class for automatic tree-building
    and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with an empty root
        and the specified depth limit."""
        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__()."""
        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):  
        """Implement the above algorithm to build
        the decision tree using the given features and
        classes to build the decision functions."""
        #TODO: finish this
        raise NotImplemented()
        
    def classify(self, features):
        """Use the fitted tree to 
        classify a list of examples. 
        Return a list of class labels."""
        class_labels = []
        #TODO: finish this
        raise NotImplemented()
        return class_labels

def load_csv(data_file_path, class_index=-1):
    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r ])
    classes= map(int,  out[:, class_index])
    features = out[:, :class_index]
    return features, classes

def generate_k_folds(dataset, k):
    #TODO this method should return a list of folds,
    # where each fold is a tuple like (training_set, test_set)
    # where each set is a tuple like (examples, classes)
    raise NotImplemented()

class RandomForest():
    """Class for random forest
    classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate, attr_subsample_rate):
        """Create a random forest with a fixed 
        number of trees, depth limit, example
        sub-sample rate and attribute sub-sample
        rate."""
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of 
        decision trees."""
        # TODO implement the above algorithm
        raise NotImplemented()

    def classify(self, features):
        """Classify a list of features based
        on the trained random forest."""
        # TODO implement classification for a random forest.
        raise NotImplemented()

class ChallengeClassifier():
    
    def __init__(self):
        # initialize whatever parameters you may need here-
        # this method will be called without parameters 
        # so if you add any to make parameter sweeps easier, provide defaults
        raise NotImplemented()
        
    def fit(self, features, classes):
        # fit your model to the provided features
        raise NotImplemented()
        
    def classify(self, features):
        # classify each feature in features as either 0 or 1.
        raise NotImplemented()