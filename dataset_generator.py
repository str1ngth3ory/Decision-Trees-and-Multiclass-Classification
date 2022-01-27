import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
'''    from sklearn.datasets import make_blobs
       from sklearn.datasets import make_gaussian_quantiles '''

plt_col = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'brown', 'coral', 'sienna', 'peru', 'orange', 'yellow', 'lime']

hand_binary = {'n_samples': 8, 'n_features': 4, 'n_informative': 4, 'n_redundant': 0, 'n_repeated': 0, 'n_classes': 2,
                'n_clusters_per_class': 1, 'weights': None, 'flip_y': 0.00, 'class_sep': 1.0, 'hypercube': True,
                'shift': 0.0, 'scale': 1.0, 'shuffle': True, 'name': 'hand_binary.csv'}
hand_multi = {'n_samples': 12, 'n_features': 4, 'n_informative': 4, 'n_redundant': 0, 'n_repeated': 0, 'n_classes': 3,
               'n_clusters_per_class': 1, 'weights': None, 'flip_y': 0.00, 'class_sep': 1.0, 'hypercube': True,
               'shift': 0.0, 'scale': 1.0, 'shuffle': True, 'name': 'hand_multi.csv'}
simple_binary = {'n_samples': 100, 'n_features': 5, 'n_informative': 4, 'n_redundant': 1, 'n_repeated': 0,
                 'n_classes': 2, 'n_clusters_per_class': 1, 'weights': None, 'flip_y': 0.00, 'class_sep': 1.0,
                 'hypercube': True, 'shift': 0.0, 'scale': 1.0, 'shuffle': True, 'name': 'simple_binary.csv'}
simple_multi = {'n_samples': 100, 'n_features': 6, 'n_informative': 4, 'n_redundant': 2, 'n_repeated': 0,
                'n_classes': 3, 'n_clusters_per_class': 1, 'weights': None, 'flip_y': 0.00, 'class_sep': 1.0,
                'hypercube': True, 'shift': 0.0, 'scale': 1.0, 'shuffle': True, 'name': 'simple_multi.csv'}
mod_complex_binary = {'n_samples': 1400, 'n_features': 7, 'n_informative': 5, 'n_redundant': 2, 'n_repeated': 0,
                      'n_classes': 2, 'n_clusters_per_class': 2, 'weights': None, 'flip_y': 0.01, 'class_sep': 0.9,
                      'hypercube': True, 'shift': 0.0, 'scale': 1.0, 'shuffle': True, 'name': 'mod_complex_binary.csv'}
mod_complex_multi = {'n_samples': 1800, 'n_features': 10, 'n_informative': 6, 'n_redundant': 2, 'n_repeated': 0,
                     'n_classes': 5, 'n_clusters_per_class': 2, 'weights': None, 'flip_y': 0.01, 'class_sep': 0.9,
                     'hypercube': True, 'shift': 0.0, 'scale': 1.0, 'shuffle': True, 'name': 'mod_complex_multi.csv'}

class DatasetGen:

    def make_ds(self, samples=0, features=0, inform=0, redund=0, repeat=0, classes=0, clstrs=2,
                wghts=None, y_flp=0., sep=1., hyper=False, shft=0., scl=1., shuff=True, seed=0.):
        '''This function will generate a simple binary dataset based on params
           This uses the sklearn dataset module to generate the dataset
            Args:
                n_samples, int, default=100 The number of samples.
                n_features, int, default=20 The total features. n_informative + n_redundant + n_repeated
                n_informative, int, default=2, informative features, composed of a number of gaussian clusters
                    each located around the vertices of a hypercube in a subspace of dimension n_informative.
                    For each cluster, informative features are drawn independently from N(0, 1) and then randomly
                    linearly combined within each cluster in order to add covariance. The clusters are then placed
                    on the vertices of the hypercube.
                n_redundant, int, default=2, redundant features generated as random linear combinations of the
                    informative features.
                n_repeated, int, default=0, duplicated features, drawn randomly from the informative
                    and the redundant features.
                n_classes, int, default=2, The number of classes (or labels) of the classification problem.
                n_clusters_per_class, int, default=2, The number of clusters per class.
                weights, array-like of shape (n_classes,) or (n_classes - 1,), default=None
                    The proportions of samples assigned to each class. If None, then classes are balanced.
                    Note that if len(weights) == n_classes - 1, then the last class weight is automatically inferred.
                    More than n_samples samples may be returned if the sum of weights exceeds 1.
                    Note that the actual class proportions will not exactly match weights when flip_y isn’t 0.
                flip_y, float, default=0.01, The fraction of samples whose class is assigned randomly. Larger values
                    introduce noise in the labels and make the classification task harder. Note that the default
                    setting flip_y > 0 might lead to less than n_classes in y in some cases.
                class_sep, float, default=1.0, The factor multiplying the hypercube size. Larger values spread out
                    the clusters/classes and make the classification task easier.
                hypercube, bool, default=True, If True, the clusters are put on the vertices of a hypercube.
                    If False, the clusters are put on the vertices of a random polytope.
                    shift, float, ndarray of shape (n_features,) or None, default=0.0, Shift features by the specified
                    value. If None, then features are shifted by a random value drawn in [-class_sep, class_sep].
                scale, float, ndarray of shape (n_features,) or None, default=1.0, Multiply features by the
                    specified value. If None, then features are scaled by a random value drawn in [1, 100]. Note
                    that scaling happens after shifting.
                shuffle, bool, default=True, Shuffle the samples and the features.
                    random_stateint, RandomState instance or None, default=None, Determines random number generation
                    for dataset creation. Pass an int for reproducible output across multiple function calls.
                Returns X: ndarray of shape (n_samples, n_features) y: ndarray of shape (n_samples,) The integer
                labels for class membership of each sample.
        '''
        ds_samples, ds_classes = make_classification(n_samples=samples, n_features=features, n_informative=inform,
                                                        n_redundant=redund, n_repeated=repeat, n_classes=classes,
                                                        n_clusters_per_class=clstrs, weights=wghts, flip_y=y_flp,
                                                        class_sep=sep, hypercube=hyper, shift=shft, scale=scl,
                                                        shuffle=shuff, random_state=seed)
        ds_classes = ds_classes.reshape((len(ds_classes), 1))
        dataset = np.concatenate((ds_samples, ds_classes), axis=1)
        return dataset

def generate_sets(ds_dict_list=[], plt_ds=False):
    ''' This generates datasets for classification problems based on SciKit-Learn 1.0.2 make_classification
        There is a minor simplification of defaults in setting some parameters
        Run this program without params to generate all of the datasets. Visualize through plt_ds
    '''
    dsg = DatasetGen()
    rand_seed = 66012022
    if not ds_dict_list:    # generate all ds and save to disk
        ds_dict_list.append(hand_binary)
        ds_dict_list.append(hand_multi)
        ds_dict_list.append(simple_binary)
        ds_dict_list.append(simple_multi)
        ds_dict_list.append(mod_complex_binary)
        ds_dict_list.append(mod_complex_multi)

    if plt_ds:
        plt.figure(figsize=(8, 8))
        plt.subplots_adjust(bottom=0.05, top=0.9, left=0.05, right=0.95)
        plt_nbr = 321
        colors = plt_col
    for ds_dict in ds_dict_list:
        dataset = dsg.make_ds(ds_dict['n_samples'], ds_dict['n_features'], ds_dict['n_informative'],
                              ds_dict['n_redundant'], ds_dict['n_repeated'], ds_dict['n_classes'],
                              ds_dict['n_clusters_per_class'], ds_dict['weights'], ds_dict['flip_y'],
                              ds_dict['class_sep'], ds_dict['hypercube'], ds_dict['shift'],
                              ds_dict['scale'], ds_dict['shuffle'], rand_seed)
        np.savetxt('./data/' + ds_dict['name'], dataset, delimiter=',')
        if plt_ds:
            plt.subplot(plt_nbr)
            plt.title(ds_dict['name'], fontsize="small")
            for cls in range(ds_dict['n_classes']):
                X = dataset[dataset[:, -1]==cls]
                for col in range(X.shape[1]-1):
                    plt.scatter(X[:, col], X[:, -1], marker='o', c=colors[col], s=20, edgecolor='k')
            plt_nbr += 1
    if plt_ds:
        plt.show()

if __name__ == '__main__':
    generate_sets(plt_ds=True)
