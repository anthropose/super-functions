# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:37:17 2019

@author: chari
"""

#%%
import warnings
import numpy as np
import pandas as pd
import scipy
import scipy.stats as st
import matplotlib.pyplot as plt

#warnings.filterwarnings("ignore")

#%%
# current scipy stats continuous distributions
scipy_distributions = [        
        st.alpha,
        st.anglit,
        st.arcsine,
        st.argus,
        st.beta,
        st.betaprime,
        st.bradford,
        st.burr,
        st.burr12,
        st.cauchy,
        st.chi,
        st.chi2,
        st.cosine,
        st.crystalball,
        st.dgamma,
        st.dweibull,
        st.erlang,
        st.expon,
        st.exponnorm,
        st.exponweib,
        st.exponpow,
        st.f,
        st.fatiguelife,
        st.fisk,
        st.foldcauchy,
        st.foldnorm,
        st.frechet_r,
        st.frechet_l,
        st.genlogistic,
        st.gennorm,
        st.genpareto,
        st.genexpon,
        st.genextreme,
        st.gausshyper,
        st.gamma,
        st.gengamma,
        st.genhalflogistic,
        st.gilbrat,
        st.gompertz,
        st.gumbel_r,
        st.gumbel_l,
        st.halfcauchy,
        st.halflogistic,
        st.halfnorm,
        st.halfgennorm,
        st.hypsecant,
        st.invgamma,
        st.invgauss,
        st.invweibull,
        st.johnsonsb,
        st.johnsonsu,
        st.kappa4,
        st.kappa3,
        st.ksone,
        st.kstwobign,
        st.laplace,
        st.levy,
        st.levy_l,
        st.levy_stable,
        st.logistic,
        st.loggamma,
        st.loglaplace,
        st.lognorm,
        st.lomax,
        st.maxwell,
        st.mielke,
        st.moyal,
        st.nakagami,
        st.ncx2,
        st.ncf,
        st.nct,
        st.norm,
        st.norminvgauss,
        st.pareto,
        st.pearson3,
        st.powerlaw,
        st.powerlognorm,
        st.powernorm,
        st.rdist,
        st.reciprocal,
        st.rayleigh,
        st.rice,
        st.recipinvgauss,
        st.semicircular,
        st.t,
        st.trapz,
        st.triang,
        st.truncexpon,
        st.truncnorm,
        st.tukeylambda,
        st.uniform,
        st.vonmises,
        st.vonmises_line,
        st.wald,
        st.weibull_min,
        st.weibull_max,
        st.wrapcauchy
        ]

#%%
# adapted from http://www.insightsbot.com/blog/WEjdW/
# fitting-probability-distributions-with-python-part-1

# define class object for distributions that has specific attributes and methods
class Distribution:
    def __init__(self, scipy_dist, data):
        """
        This object has a scipy-related continuous distribution and 
        user-inputted data.
        """
        self.SciPyDistribution = scipy_dist
        self.name = scipy_dist.name
        self.Params = {}
        self.PValue = 0
        self.PDFParams = {}
        self.SSE = np.inf
        
        # if user data is greater than zero, go ahead and fit the data to all
        # possible scipy distributions and calculate the SSE associated with 
        # each fit
        if len(data) > 0:
            self.Fit(data)
            self.CalculateSSE(data, len(data))
   

     
    def Fit(self, data):
        """
        This class method gets the associated attributes (i.e., name and 
        parameters) of a scipy continuous distribution and uses maximum 
        likelihood to estimate the distribution parameters (including location 
        and scale) of observed data in comparison to the scipy distribution. 
        Presumably it does this by minimizing the negative log-likelihoods.
        """        
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
            
            # use the getattr method to obtain associated statistics and name
            # of a scipy distribution
            dist = getattr(scipy.stats, self.SciPyDistribution.name)
            
            # fit the scipy distribution to observed data using ML and save the
            # estimated parameters
            self.Params = dist.fit(data)
                
            # apply the Kolmogorov-Smirnov test for goodness of fit between
            # scipy distribution and observed data
            D, p = scipy.stats.kstest(data, 
                                      self.SciPyDistribution.name, 
                                      args=self.Params);
            
            # set the pvalue for this distribution
            self.PValue = p
        except Exception:
           pass
    

    
    def CalculateSSE(self, data, bins = 50, ax=None):
        """
        This class method calculates the sum of squared error between a fitted
        pdf using a scipy continuous distribution and observed data
        """
        # get histogram of original data
        y, x = np.histogram(data, bins = bins, density = True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit distribution to data using scipy continuous distributions
                # and store the estimated parameters
                # fit is performed using maximum likelihood to estimate 
                # parameters of observed distribution??
                params = self.SciPyDistribution.fit(data)

                # separate parts of estimated parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # calculate fitted PDF and error with fit in distribution
                # .pdf method takes in statistics about scipy continuous
                # distribution and generates a pdf for any value x
                pdf = self.SciPyDistribution.pdf(x, 
                                                 loc = loc, 
                                                 scale = scale, 
                                                 *arg)
                
                # find the sum of squared error between the pdf at value x and
                # the observed value
                sse = np.sum(np.power(y - pdf, 2.0))

                # to facilitate plotting in plot method of class
                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax = ax)
                except Exception:
                    pass

                # generates PDF parameters and associated SSEs
                self.PDFParams = params
                self.SSE = sse
    
        except Exception:
           pass



        
    def Plot(self, data):
        """
        This class method generates histograms of fitted scipy continous
        distributions and observed data on the same plot
        """
        try:
            dist = getattr(scipy.stats, self.SciPyDistribution.name)
            
            distPlt = dist.rvs(
                    *self.Params[:-2], 
                    loc = self.Params[-2], 
                    scale = self.Params[-1], 
                    size = len(data)
                    )
            
            fig=plt.figure()
            fig.suptitle("%s Plot Showing Fitted vs Observed Data (p = %f)" 
                         % (self.SciPyDistribution.name, self.PValue))
            plt.hist(distPlt, alpha = 0.5, label = 'Fitted')
            plt.hist(data, alpha = 0.5, label = 'Observed')
            plt.legend(loc = 'upper right')
        except:
            print("Unable to plot for %s distribution" 
                  % round(self.SciPyDistribution.name, 5))
            



            
    def PlotPDF(self, data, size = 10000, xsize = 8, ysize = 6):
        """ 
        This class method generates a plot of the pdf for scipy continous
        distributions along with a histogram of the observed data. Adapted from
        https://stackoverflow.com/questions/6620471/
        fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
        """
        # separate parts of parameters
        arg = self.PDFParams[:-2]
        loc = self.PDFParams[-2]
        scale = self.PDFParams[-1]
    
        # set sane start and end points of distribution
        start = self.SciPyDistribution.ppf(0.01, *arg, loc = loc, scale = scale) if arg else self.SciPyDistribution.ppf(0.01, loc = loc, scale = scale)
        end = self.SciPyDistribution.ppf(0.99, *arg, loc = loc, scale = scale) if arg else self.SciPyDistribution.ppf(0.99, loc = loc, scale = scale)
    
        # build PDF and turn into pandas Series
        x = np.linspace(start, end, size)
        y = self.SciPyDistribution.pdf(x, loc = loc, scale = scale, *arg)
        pdf = pd.Series(y, x)
        
        # display
        plt.figure(figsize=(xsize, ysize))
        ax = pdf.plot(lw = 2, label = 'PDF', legend = True)
        data.plot(kind = 'hist', 
                  bins = 50, 
                  density = True, 
                  alpha = 0.5, 
                  label = 'Data', 
                  legend = True, 
                  ax = ax)
        
        param_names = (self.SciPyDistribution.shapes + ', loc, scale').split(', ') if self.SciPyDistribution.shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, self.PDFParams)])
        dist_str = '{}({})'.format(self.PDFParams, param_str)
        
        ax.set_title(('PDF of %s \n' % self.name) + dist_str)
        ax.set_xlabel('Histogram of Observed Data')
        ax.set_ylabel('Probability')


#%%
# define a function to get the best candidate distributions that were fit to 
# the observed data
# calls the Distributions class
def GetDistributions(data, 
                     dist_list = [st.norm,st.lognorm,st.expon], 
                     min_pvalue = 0.05, 
                     reverse = True):
    """
    This function calls the Distribution class's main functions for fit and
    calculating SSE, and returns a list of candidate distributions sorted by
    their p-values for the K-S test statistic. The K-S test statistic measures 
    the largest distance between the empirical distribution function of the 
    observed data and the cdf of the theoretical/candidate function.
    
    H0: observed data has the same underlying distribution as the candidate 
    distribution
    Ha: observed data does not have the same underlying distribution
    """
    
    distributions = []
    
    for dist in dist_list:
            
        new_distribution = Distribution(dist, data)
        
        if new_distribution.PValue > min_pvalue:
            distributions.append(new_distribution)
        
    return sorted(distributions, key = lambda x: x.PValue, reverse = reverse)

#%%
def DistributionsByColumn(data):
    """
    This function takes in a dataframe and list of column names and loops over
    the GetDistributions function from distfuncs.py. Returns a dictionary of 
    column names (i.e., groups) with their associated candidate distributions,
    parameters, and SSEs
    """
    # create list of column names
    columns = data.columns.values.tolist()
    # initialize empty dictionary that will hold column names as keys with 
    # lists of distribution objects as their values
    column_distributions = {}
    
    # loop over the GetDistributions function
    for name in columns:
        # find distribution, parameters, and SSEs
        distributions = GetDistributions(data[name], scipy_distributions)
        column_distributions[name] = distributions
        print("Calculated [%d] Distributions for Column [%s]" % (len(distributions), name))
        
    return column_distributions

#%%
# alternative approach?
# generates maximum likelihood estimates for each model, but I'm not sure for 
# which parameter of that model
# technically, the fit method in the other approaches uses ML somehow
# goal is to find the model with the smallest negative log-likelihood; in other
# words, goal is to minimize the negative of the log-likelhood

#max_like_est = {}
#
#for distribution in scipy_distributions:
#        pars = distribution.fit(data)
#        mle = distribution.nnlf(pars, data)
#        max_like_est[distribution] = mle
