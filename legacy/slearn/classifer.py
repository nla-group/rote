# Copyright (c) 2021, nla group, manchester
# All rights reserved. 

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import re
import copy
import random
import warnings
import numpy as np
import pandas as pd
# import lightgbm as lgb
from .symbols import *

from sklearn.svm import SVC
# from deepforest import CascadeForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import warnings
import os            


class symbolicML:
    """Classifier for symbolic next-symbol prediction.

    Parameters
    ----------
    classifier_name : str, default='MLPClassifier'
        Name of the scikit-learn estimator used for next-symbol prediction.
        Common choices include ``MLPClassifier``, ``KNeighborsClassifier``,
        ``GaussianProcessClassifier``, ``QuadraticDiscriminantAnalysis``,
        ``DecisionTreeClassifier``, ``LogisticRegression``,
        ``AdaBoostClassifier``, ``GaussianNB``, and ``SVC``.
    ws : int, default=3
        Context-window length used to construct fixed-dimensional symbolic
        features. Larger windows expose more history to the classifier.
    random_seed : int, default=0
        Random seed forwarded to compatible scikit-learn estimators.
    verbose : int, default=0
        Verbosity level. A value of zero suppresses progress output where the
        wrapped estimator supports it.
    """
        
    def __init__(self, classifier_name='MLPClassifier', ws=3, random_seed=0, verbose=0):

        os.environ['PYTHONHASHSEED']=str(42)

        self.classifier_name = classifier_name 
        self.init_classifier()
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        self.verbose = verbose
        
        self.ws = ws
        self.mu = 0
        self.scl = 1
        


    @property
    def random_seed(self):
        return self._random_seed



    @random_seed.setter
    def random_seed(self, value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError('Expected int type.')
        
        if value < 0:
            raise ValueError(
                "Please feed an correct value for random_seed.")
        
        self._random_seed = value
        


    @property
    def ws(self):
        return self._ws



    @ws.setter
    def ws(self, value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError('Expected a float or int type.')
        
        if value < 0:
            raise ValueError(
                "Please feed an correct value for ws.")
        
        if value == 1:
            warnings.warn("There is no dimensionaity reduction in symbolic representation.")
        
        self._ws = value
            
            
            
    @property
    def verbose(self):
        return self._verbose 



    @verbose.setter
    def verbose(self, value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError('Expected int numeric type.')
        
        if value < 0:
            raise ValueError(
                "Please feed an correct value for verbose.")
        
        self._verbose = value
        


    def encode(self, string):
        """
        Construct features and target labels for symbols and encode to numerical values.

        Parameters
        ----------
        string: {str, list}
            symbolic string.
        
        """
        if not isinstance(string, list):
            string_split = [s for s in re.split('', string) if s != '']
        else:
            string_split = copy.deepcopy(string)
        
        self.hashm = dict(zip(set(string_split), np.arange(len(set(string_split)))))
        string_encoding = [self.hashm[i] for i in string_split] 
        
        if self.ws > len(string_encoding):
            warnings.warn("ws is larger than the series, please reset the ws.")
            self.ws = len(string_encoding) - 1
        x, y = self.construct_train(string_encoding)
        
        return x, y
        
    
    def construct_train(self, series):
        """
        Construct features and target labels for symbols.

        Parameters
        ----------
        series - numpy.ndarray:
            The numeric time series. 
        
        """


        features = list()
        targets = list()
        for i in range(len(series) - self.ws):
            features.append(series[i:i+self.ws])
            targets.append(series[i+self.ws])
        return np.array(features), np.array(targets)



    def init_classifier(self):
        if self.classifier_name == "KNeighborsClassifier":
            self.Classifiers = KNeighborsClassifier
        
        elif self.classifier_name == "GaussianProcessClassifier":
            self.Classifiers = GaussianProcessClassifier
        
        elif self.classifier_name == "QuadraticDiscriminantAnalysis":
            self.Classifiers =  QuadraticDiscriminantAnalysis
        
        elif self.classifier_name == "DecisionTreeClassifier":
            self.Classifiers = DecisionTreeClassifier
        
        elif self.classifier_name == "LogisticRegression":
            self.Classifiers = LogisticRegression
        
        elif self.classifier_name == "AdaBoostClassifier":
            self.Classifiers = AdaBoostClassifier
        
        elif self.classifier_name == "RandomForestClassifier":
            self.Classifiers = AdaBoostClassifier
        
        elif self.classifier_name == "GaussianNB":
            self.Classifiers = GaussianNB
        
        # elif self.classifier_name == "DeepForest":
        #    self.Classifiers = CascadeForestClassifier
        
        # elif self.classifier_name == "LGBM":
            # lgb_params = {'boosting_type': 'gbdt',
            #          'learning_rate': 0.5,
            #          'max_depth': 5
            #          }
        #    self.Classifiers = lgb.LGBMClassifier
        
        elif self.classifier_name == "SVC":
            self.Classifiers = SVC
        
        elif self.classifier_name == "RBF":
            self.Classifiers = RBF

        else: # "MLPClassifier"
            self.Classifiers = MLPClassifier
         


    def forecast(self, x, y, step=5, inversehash=None, centers=None, **params):
        try:
            cparams = copy.deepcopy(params)
            if "verbose" in self.Classifiers().__dict__:
                if not self.verbose:
                    cparams['verbose'] = 0
            if "random_state" in self.Classifiers().__dict__:
                cparams['random_state'] = 0
            clf = self.Classifiers(**cparams)
            clf.fit(x, y)
        except:
            # warnings.warn("fail to set_random_state.")
            params.pop('random_state', None)
            
            clf = self.Classifiers(**params)
            clf.fit(x, y)

        if inversehash == None:
            for i in range(step):
                last_x = np.hstack((x[-1][1:], y[-1]))
                pred = clf.predict(np.expand_dims(last_x, axis=0))
                x = np.vstack((x, last_x))
                y = np.hstack((y, pred))
            inversehash = dict(zip(self.hashm.values(), self.hashm.keys()))
            symbols_pred = [inversehash[n] for n in y[-step:]]
        else:
            for i in range(step):
                last_x = np.hstack((x[-1][1:], (centers[y[-1]] - self.mu)/self.scl))
                pred = clf.predict(np.expand_dims(last_x, axis=0))
                x = np.vstack((x, last_x))
                y = np.hstack((y, pred))
            symbols_pred = [inversehash[n] for n in y[-step:]]
        return symbols_pred


    
    

class slearn(symbolicML):
    """Symbolic time-series forecasting wrapper.

    Parameters
    ----------
    method : {'SAX', 'ABBA', 'fABBA'}, default='fABBA'
        Symbolic representation used before classifier training.
    ws : int, default=1
        Context-window length used by the symbolic classifier.
    step : int, default=10
        Number of future symbols, or reconstructed numeric samples, to predict.
    classifier_name : str, default='MLPClassifier'
        Name of the scikit-learn estimator used by ``symbolicML``.
    form : {'numeric', 'string'}, default='numeric'
        Return reconstructed numeric forecasts or symbolic forecasts.
    random_seed : int, default=0
        Random seed used by compatible estimators.
    verbose : int, default=1
        Verbosity level for symbolic transformation and classifier training.
    """
        
    def __init__(self, method='fABBA', ws=1, step=10, 
                 classifier_name="MLPClassifier",
                 form='numeric', random_seed=0, verbose=1):
        
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        self.ws = ws
        self.classifier_name = classifier_name
        self.verbose = verbose
        self.form = form
        self.step = step
        self.method = method
        self.params_secure()
        


    @property
    def random_seed(self):
        return self._random_seed



    @random_seed.setter
    def random_seed(self, value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError('Expected int type.')
        
        if value < 0:
            raise ValueError(
                "Please feed an correct value for random_seed.")
        
        self._random_seed = value
        


    @property
    def step(self):
        return self._step



    @step.setter
    def step(self, value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError('Expected a float or int type.')
        
        if value < 0:
            raise ValueError(
                "Please feed an correct value for step.")
        
        self._step = value
            
            
            
    @property
    def ws(self):
        return self._ws



    @ws.setter
    def ws(self, value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError('Expected a float or int type.')
        
        if value < 0:
            raise ValueError(
                "Please feed an correct value for ws.")
        
        if value == 1:
            warnings.warn("There is no dimensionaity reduction in symbolic representation.")
        
        self._ws = value
            
            
            
    @property
    def verbose(self):
        return self._verbose 



    @verbose.setter
    def verbose(self, value):
        if not isinstance(value, float) and not isinstance(value,int):
            raise TypeError('Expected int numeric type.')
        
        if value < 0:
            raise ValueError(
                "Please feed an correct value for verbose.")
        
        self._verbose = value
            
            
            
    def set_symbols(self, series, **kwargs):
        """Transform time series to specified symplic representation
        
        Please feed into the parameters for the corresponding symbolic representation.

        Parameters
        ----------
        series - numpy.ndarray:
            The numeric time series. 
            
        """

        
        if not isinstance(series, np.ndarray):
            series = np.array(series)
            
        self.mu = series.mean()
        self.scl = series.std()
        
        if self.scl == 0:
            self.scl = 1
        scale_series = (series - self.mu) / self.scl
        
        self.start = scale_series[0]
        self.length = len(series)

        if self.method == 'fABBA':
            try:
                self.s_model = fABBA(**kwargs, verbose=self.verbose)  
            except:
                warnings.warn("Exception, default setting (tol=0.1, alpha=0.1, sorting='2-norm') apply.")
                self.s_model = fABBA(tol=0.1, alpha=0.1, sorting='2-norm', verbose=self.verbose)
            
            self.string = self.s_model.fit_transform(scale_series)         
            self.last_symbol = self.string[-1] # deprecated symbol, won't take into account
                                            # only apply to ABBA
        
        elif self.method == 'ABBA':
            try:
                self.s_model = ABBA(**kwargs, verbose=self.verbose)
            except:
                warnings.warn(f"Exception, default setting (tol=0.1, k_cluster=2, apply.")
                self.s_model = ABBA(tol=0.1, k_cluster=2, verbose=self.verbose)
                
            self.string = self.s_model.fit_transform(scale_series)         
            self.last_symbol = self.string[-1] # deprecated symbol, won't take into account
                                            # only apply to ABBA
                
        elif self.method == 'SAX':
            try:
                if 'n_paa_segments' in kwargs:
                    kwargs['width'] = self.length // kwargs['n_paa_segments']
                    del kwargs['n_paa_segments']
                self.s_model = SAX(**kwargs, verbose=self.verbose, return_list=True)
            except:
                # kwargs['n_paa_segments'] = 10
                # width = self.length // kwargs['n_paa_segments']
                # self.s_model = SAX(width=width, k=kwargs['k'], return_list=True)
                warnings.warn("Exception, width for SAX is set to 1.")
                self.s_model = SAX(width=1, k=self.length, return_list=True)
            self.string = self.s_model.transform(scale_series)
            
        else:
            raise ValueError(
                "Sorry, there is no {} method for now. Will use the 'fABBA' method with default settings.".format(self.method))
            
        if self.ws >= len(self.string):
            warnings.warn("Parameters are not appropriate, classifier might not converge.")
            warnings.warn("Degenerate to trivial case that ws=1.")
            self.ws = 1

        return
    

    def predict(self, **params):
        self.cmodel = symbolicML(classifier_name=self.classifier_name,
                          ws=self.ws, 
                          random_seed=self.random_seed
                         )
        if self.verbose:
            print("-------- Config --------")
            print("The length of time series: ", self.length)
            print("The number of symbols: ", len(self.string))
            print("The dimension of features is: ", self.ws)
            print("The number of symbols to be predicted: ", self.step)
            print("The parameters of classifiers: ", params)
        
        if self.method == 'fABBA' or self.method == 'ABBA':
            x, y = self.cmodel.encode(self.string[:-1]) # abandon the last symbol
        else:
            x, y = self.cmodel.encode(self.string)
        
        if 'random_state' not in params:
            params['random_state'] = self.random_seed
            
        if self.form == 'string':
            return self.cmodel.forecast(x, y, step=self.step, **params)
        else:
            pred = self.cmodel.forecast(x, y, step=self.step, **params)
            if self.method == 'SAX':
                inverse_ts = self.s_model.inverse_transform(self.string+pred)
            else:
                inverse_ts = self.s_model.inverse_transform(self.string[:-1]+pred, self.start)
                
            inverse_ts = np.array(inverse_ts) * self.scl + self.mu
            return inverse_ts[self.length:]
    


    def params_secure(self):
        """Check parameter settings"""
        
        if not isinstance(self.method, str):
            raise ValueError("Please ensure method is string type!")
            
        if not (isinstance(self.random_seed, float) or isinstance(self.random_seed, int)):
            raise ValueError("Please ensure random_seed is numeric type!")
            
        if (not isinstance(self.ws, int)) and self.ws > 0:
            raise ValueError("Please ensure ws is integer!")
            
        if (not isinstance(self.step, int)) and self.step > 0:
            raise ValueError("Please ensure ws is integer!")
            
        if not isinstance(self.classifier_name, str):
            raise ValueError("Please ensure classifier_name is string type!")

            
            
            
