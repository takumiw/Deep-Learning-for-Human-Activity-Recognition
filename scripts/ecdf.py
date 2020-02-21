# -*- coding:utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np


class MultivariateFeatureExtract(object):
    """This is base class for feature extraction from multivariate sequences.
    Attributes:
        params: dictionary type properties which containe optional parameters for feature extraction
        base_data: training data for some feature extraction, such as DistanceBased.
          For example, in DistanceBased feature extraction, the distance between input multivariate-sequences
          and self.base_data will be return. The property is None at initialize stage.
        UNITE_TYPE: Default unite type
    """
    __metaclass__ = ABCMeta
    UNITE_TYPE = "Concat"
    def __init__(self, **params):
        """Inits BaseFeatureExtract."""
        self.params = params
        self.base_data = None
    def show_params(self):
        """Show optional parameters of this class."""
        print(self.params)
    def set_basedata(self, multi_sequences, label=None):
        if self._validate(multi_sequences):
            self.base_data = multi_sequences
        else:
            raise Exception('The type of input args strems is invalid')
        return self
    def transform(self, multi_sequences, unite_type=None):
        # Validate multi_sequences
        #   1. validate the input multi_sequences itself
        if self._validate(multi_sequences):
            pass
        else:
            raise Exception(
                'The type of input args multi_sequences is invalid'
            )
        # print "The number of sequence: %d" % len(multi_sequences)
        #   2. validate base_data is set
        if self.base_data is None:
            raise Exception(
                'To excec transfer(), you need to set base_data using set_basedata() earlier')
        #   3. validate the number of sequence is same between multi_sequences and self.base_data
        if len(multi_sequences) != len(self.base_data):
            raise Exception(
                '\nThe length of input multi_sequences is %d and ' % len(multi_sequences)
                + '\nthe length of input self.base_data is %d and ' % len(self.base_data)
                + '\nThe number of sequence must be same with self.base_data'
                '!!!')
        # set default unite_type
        unite_type = self.UNITE_TYPE if unite_type is None else unite_type
        feature_list = []
        for i in range(0, len(multi_sequences)):
            feature_list.append(self._do(
                multi_sequences[i], self.base_data[i]))
        return self._unite_features(feature_list, unite_type)
    def _unite_features(self, features, unite_type):
        """Union the features into one feature
        In main method '_do' convert each sequences into a feature matrix.
        This '_union_feature' method unite feature matrixes into a feature matrix.
        ***Example
        >>> features = [np.array([0, 1], [2, 3]), np.array([0, 1], [2, 3])]
        >>> self._unite_features(features)
        [np.array([0, 2], [4, 6])]
        Args:
            features: each element feature[i] indicate the feature matrix converted from
              input maltivariate-sequences[i].
            unite_type: difine how to convert some feature matrixes into a feature matrix.
              It could take {'Add', 'Concat'}
        Returns:
            united_feature: united feature matrix
        Raises:
        """
        if unite_type not in {'Add', 'Concat'}:
            raise ValueError('unite type mus be \'Add\' or \'Concat\'')
        # else:
        #     print "Unite Type: %s" % unite_type
        united_feature = features[0]
        if unite_type == 'Add':
            for i in range(1, len(features)):
                # print "Add feature[i]"
                # print "Before: %s" % str(united_feature)
                united_feature = united_feature + features[i]
                # print "After: %s" % str(united_feature)
            return united_feature
        if unite_type == 'Concat':
            return np.hstack(features) 
            # for i in range(1, len(features)):
            #     # print "Concat feature[i]"
            #     # print "Before: %s" % str(united_feature)
            #     united_feature = np.hstack([united_feature, features[i]])
            #     # print "After: %s" % str(united_feature)
            # return united_feature
    @abstractmethod
    def _do(self, sq1, sq2):
        pass
    @abstractmethod
    def _validate(self, multi_sequences):
        pass
    def get_param(self, key):
        if key in self.params.keys():
            return self.params[key]
        else:
            return self.DEFAULT_PARAMS[key]

class _NumericSequenceFE(MultivariateFeatureExtract):
    """This is base class for feature extraction
    from multivariate-numeric sequences.
    This class implement a method '_validate', a abstractmethod of Super Class.
    The method '_validate' make sure the all of input multivariate-sequences is
    numeric sequences.
    Attributes:
    """
    __metaclass__ = ABCMeta
    def _validate(self, multi_sequences):
        return all(map(lambda x: _validate_is_numericsq(x), multi_sequences))

def ecdf(x, n, whiten=False):
    y = x.copy()
    # if whiten is True:
    #     y = (y - y.mean())/y.std()
    y.sort()
    p = 1. * np.arange(len(y)) / (len(y) - 1)
    p_val = np.linspace(0, 1, n)
    return np.interp(p_val, p, y)

class ECDF(_NumericSequenceFE):
    DEFAULT_PARAMS = {'whiten': True, 'n_bins': 10}
    """This class provide the functions for distance-based feature extraction.
    This class calculates the distance between self.base_data and
    input multivariate-sequences.
    ** Example
    >>> base_data = [[1,1], [1,1]]
    >>> test_data = [[1,1], [1,1]]
    >>> feature_extract = DistanceBased()
    >>> feature_extract.set_basedata(base_data)
    >>> feature = feature_extract.transform(test_data)
    [0]
    Attributes:
    """
    UNITE_TYPE = "Concat"

    #def _do(self, sq1, sq2):
    def _do(self, sq1):
        if 'n_bins' in self.params.keys():
            n_bins = self.params['n_bins']
        else:
            n_bins = self.DEFAULT_PARAMS['n_bins']
        if 'whiten' in self.params.keys():
            whiten = self.params['whiten']
        else:
            whiten = self.DEFAULT_PARAMS['whiten']
        _ecdf = np.array([ecdf(ts, n_bins, whiten=whiten) for ts in sq1])
        return _ecdf
