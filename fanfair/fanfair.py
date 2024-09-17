import simpful as sf
import pandas as pd
import numpy as np
from numpy import array, sqrt, abs, median, arange, argmax
from statistics import stdev
from math import prod
from scipy import stats
import matplotlib.pyplot as plt
from math import pi

import warnings #TODO: solve isotree warnings

class FanFAIR:

  def __init__(self, dataset=None, dataframe=None, force_csv=False, output_column=None,
               drop_columns=None, outliers_detection_method="ECOD", balance_method="Hellinger"):

    # default values
    self._balance_value = None
    self._numerosity_value = None
    self._outliers_ratio = None
    self._compliance_value = None
    self._incompleteness = None
    self._numsamples = 0
    self._numfeatures = 0
    self._maxsensitivity = 0

    self._sensitive_variables = []
    self._worst_correlation_name = None
    self._input_dataframe = None
    self._column_names = None
    self._clean_dataframe = None
    self._dataset_file = None

    # if a dataset is specified, open the file with pandas and extract info
    if dataset is not None:
      dataframe = self._import_dataset_fromfile(dataset, force_csv)
      self._import_dataset(dataframe, output_column, drop_columns, \
                           outliers_detection_method=outliers_detection_method, balance_method=balance_method)
    elif dataframe is not None:
      self._import_dataset(dataframe, output_column, drop_columns, \
                           outliers_detection_method=outliers_detection_method, balance_method=balance_method)
    else:
      raise Exception("Please specify a dataset as file or Pandas' dataframe.")


    # create the fuzzy reasoner for dataset assesment
    self._reasoner = sf.FuzzySystem(verbose=True, show_banner=False)

    # create linguistic variable "balance"
    LV_balance = sf.AutoTriangle(2, terms=["low", "high"], universe_of_discourse=[0,1])
    self._reasoner.add_linguistic_variable("balance", LV_balance)

    # create linguistic variable "numerosity"
    LV_numerosity = sf.AutoTriangle(2, terms=["low", "high"], universe_of_discourse=[0,10])
    self._reasoner.add_linguistic_variable("numerosity", LV_numerosity)

    # create linguistic variable "unevenness"
    LV_outliers = sf.AutoTriangle(2, terms=["low", "high"], universe_of_discourse=[0,0.1])
    self._reasoner.add_linguistic_variable("unevenness", LV_outliers)

    # originally, compliance ranged from 0 to 10 and considered:
    # anon, license, legal basis, DPIA, LIA, transfer, reuse, contract, explicit consent, ethical assessment
    # the current version of FanFAIR assumes 5 criteria:
    # 1) data protection law
    # 2) copyright law
    # 3) medical law
    # 4) non-discrimination law
    # 5) ethics
    LV_compliance = sf.AutoTriangle(2, terms=["low", "high"], universe_of_discourse=[0,5])
    self._reasoner.add_linguistic_variable("compliance", LV_compliance)

    # create linguistic variable "quality"
    LV_quality = sf.AutoTriangle(2, terms=["low", "high"], universe_of_discourse=[0,1])
    self._reasoner.add_linguistic_variable("quality", LV_quality)

    # create linguistic variable "incompleteness"
    LV_incompleteness = sf.AutoTriangle(2, terms=["low", "high"], universe_of_discourse=[0,1])
    self._reasoner.add_linguistic_variable("incompleteness", LV_incompleteness)

   # create linguistic variable "correlation" between sensitive variable(s) and output
    LV_correlation = sf.AutoTriangle(2, terms=["low", "high"], universe_of_discourse=[0,1])
    self._reasoner.add_linguistic_variable("correlation", LV_correlation)

    # create outputs
    self._reasoner.set_crisp_output_value("low_fairness", 0)
    self._reasoner.set_crisp_output_value("high_fairness", 1)

    # create rule base
    R1 = "IF (balance IS high) THEN (phi IS high_fairness)"
    R2 = "IF (balance IS low) THEN (phi IS low_fairness)"

    R3 = "IF (numerosity IS high) THEN (phi IS high_fairness)"
    R4 = "IF (numerosity IS low) THEN (phi IS low_fairness)"

    R5 = "IF (unevenness IS high) THEN (phi IS low_fairness)"
    R6 = "IF (unevenness IS low) THEN (phi IS high_fairness)"

    R7 = "IF (compliance IS high) THEN (phi IS high_fairness)"
    R8 = "IF (compliance IS low) THEN (phi IS low_fairness)"

    R9 = "IF (quality IS high) THEN (phi IS high_fairness)"
    R10 = "IF (quality IS low) THEN (phi IS low_fairness)"

    R11 = "IF (incompleteness IS high) THEN (phi IS low_fairness)"
    R12 = "IF (incompleteness IS low) THEN (phi IS high_fairness)"

    # NEW for sensitive variables
    R13 = "IF (correlation IS high) THEN (phi IS low_fairness)"
    R14 = "IF (correlation IS low) THEN (phi IS high_fairness)"

    self._reasoner.add_rules([R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14])


  def set_sensitive_variables(self, list_sensitive_variables):
    """
      FanFAIR 2 gives the possibility to specify some 'sensitive' variables that
      should never correlate with the output (else, some discrimination is present
      in the dataset).

      The absolute value of each sensitive variable is computed and aggregated using 
      fuzzy operators.
    """

    self._sensitive_correlations = {}

    if self._column_names is None:
      raise Exception("cannot set sensitive variable(s) without opening a dataset first, aborting.")
    
    for sens_variable in list_sensitive_variables:
      if sens_variable not in self._column_names:
        raise Exception("the sensitive variable '%s' is not present in the dataset, aborting." % sens_variable)
      else:

        # currently, only binary labels are supported
        values = set(self._output_dataframe)
        print(" * Detected %d levels in output:" % len(values), values)
        if len(values)!=2:
          raise Exception("not supported, aborting.")
        new_values = self._output_dataframe.replace(to_replace=values, value=[0,1])
        abs_correlation = abs(self._input_dataframe[sens_variable].corr(new_values))
        self._sensitive_correlations[sens_variable] = abs_correlation
        print(" * Absolute correlation value between sensitive variable '%s' and output: %.2f" % (sens_variable, abs_correlation))
    
    worst_corr = max(self._sensitive_correlations.values())
    worst_name = max(self._sensitive_correlations, key=self._sensitive_correlations.get)
    
    self._set_maxsensitivity(worst_corr)
    self._worst_correlation_name = worst_name
    
    self._sensitive_variables = list_sensitive_variables
    print(" * Sensitive variable(s) set:", ", ".join(self._sensitive_variables))


  def _import_dataset_fromfile(self, path, force_csv=False):
    """
      Imports the dataset as Panda's dataframe from path. Only supports XLSX and CSV files.
    """
    if path[-3:]=="csv" or force_csv:
      DF = pd.read_csv(path).reset_index(drop=True)
    elif path[-4:]=="xlsx":
      DF = pd.read_excel(path).reset_index(drop=True)
    else:
      raise Exception("Not supported file type.")
    self._dataset_file = path
    return DF

  def _import_dataset(self, DF, output_column, drop_columns=None, outliers_detection_method='ECOD', balance_method="sigma_ratio"):

    # partition data into input and output
    #DF = pd.read_csv(path).reset_index(drop=True)
    if drop_columns is not None:
      DF.drop(drop_columns, inplace=True, axis=1)

    self._clean_dataframe = DF.copy()

    input_DF = DF.drop([output_column], axis=1)
    output_DF = DF[output_column]

    # experimental
    self._column_names = input_DF.columns

    # check that we have some data!
    self.set_numsamples(len(DF))
    if self._numsamples==0:
      print("WARNING: empty dataset detected")
    else:
      print(" * Number of samples: %d" % self._numsamples)

    # check number of features
    self.set_numfeatures(len(input_DF.columns))
    if self._numfeatures==0:
      print("WARNING: no features detected in the dataset")
    else:
      print(" * Number of features: %d" % self._numfeatures)

    # check NaNs
    total_nans = input_DF.isna().sum().sum()
    total_values = prod(input_DF.shape)
    ratio_nans = total_nans/total_values
    print(" * %d/%d NaN values in the dataframe detected (%.2f%%)" % (total_nans, total_values, ratio_nans))
    self.set_incompleteness(ratio_nans)


    ######################
    # OUTLIERS DETECTION #
    ###################### 
    
    outliers = {}

    self._used_outlier_method = outliers_detection_method

    if outliers_detection_method=="zscores":
      for column in input_DF.columns:
        zscores = abs(stats.zscore(DF[column]))
        outliers[column] = len(DF[column][zscores>=3])

      print(" * Outliers detected (anomalous Z-score):")
      print(outliers)
      total_outliers = sum(outliers.values())
      ratio_outliers_samples = total_outliers/total_values
      print(" * Ratio outliers:total values: %.2f" % ratio_outliers_samples)
      self.set_outliers(ratio_outliers_samples)

    elif outliers_detection_method=="modified_zscores":
      for column in input_DF.columns:

        array_median = median(DF[column])
        abs_diff = abs(DF[column]-array_median)
        median_abs_diff = median(abs_diff)
        modified_zscores =  0.6745* (DF[column]-array_median)/median_abs_diff
        #print(modified_zscores)
        outliers[column] = len(DF[column][modified_zscores>=3])

      print(" * Outliers detected (anomalous modified Z-score):")
      print(outliers)
      total_outliers = sum(outliers.values())
      ratio_outliers_samples = total_outliers/total_values
      print(" * Ratio outliers:total values: %.2f" % ratio_outliers_samples)
      self.set_outliers(ratio_outliers_samples)
      
    elif outliers_detection_method=="ECOD":
      from pyod.models.ecod import ECOD
      print(" * Detecting multivariate outlying objects with ECOD...")
      clf = ECOD(n_jobs=-1) # n_jobs=-1 -> use all cores
      clf.fit(input_DF, output_DF)
      total_outlying_objects = sum(clf.labels_)
      ratio_outlying_objects = total_outlying_objects/len(DF)
      print(" * Calculated outlying instances: %d/%d (%.2f%%)" % (total_outlying_objects, len(DF), ratio_outlying_objects*100))
      self.set_outliers(ratio_outlying_objects)

    elif outliers_detection_method=="IForest":
      from pyod.models.iforest import IForest
      print(" * Detecting multivariate outlying objects with Isolation Forest...")
      clf = IForest()
      clf.fit(input_DF, output_DF)
      total_outlying_objects = sum(clf.labels_)
      ratio_outlying_objects = total_outlying_objects/len(DF)
      print(" * Calculated outlying instances: %d/%d (%.2f%%)" % (total_outlying_objects, len(DF), ratio_outlying_objects*100))
      self.set_outliers(ratio_outlying_objects)

    elif outliers_detection_method=="isotree":
      from isotree import IsolationForest

      # TODO: notify that all numbers are going float and dates are being excluded

      X = input_DF.astype({ccc:'float' for ccc in input_DF.select_dtypes('number')}).select_dtypes(exclude='datetime64[ns]')

      # TODO: UserWarning: Instantiating CategoricalDtype without any arguments?
      with warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)
        ol_ev = IsolationForest().fit(X)

      ol_scores = ol_ev.predict(X)

      # TODO: provide other options for determining OL
      # TODO: magic numbers! (.7 min tresh, 3 std)
      outliers = ol_scores > min(.7,np.mean(ol_scores) + 3 * np.std(ol_scores))

      total_outliers = outliers.sum()
      ratio_outliers_samples = total_outliers/total_values
      print(" * Calculated outlying instances (Isotree): %d/%d (%.2f%%)" % (total_outliers, len(DF), ratio_outliers_samples*100))
      self.set_outliers(ratio_outliers_samples)

    else:
      raise Exception(" * %s outlier detection method not supported, aborting." % outliers_detection_method)


    #########################
    # NUMEROSITY ASSESSMENT #
    #########################


    # calculate numerosity ratio (above 10 is fine)
    num_ratio = self._numsamples/self._numfeatures
    self.set_numerosity(num_ratio)


    ######################
    # BALANCE ASSESSMENT #
    ######################

    self._used_balance_method = balance_method

    # calculate balance
    if balance_method=="sigma_ratio":
      print(output_DF.value_counts())
      counts_classes = array(output_DF.value_counts())
      normalized_counts = counts_classes/counts_classes.sum()
      print(" * Detected classes proportions:", normalized_counts)
      std_normalized_counts = stdev(normalized_counts)
      ref_std = [0]*len(counts_classes); ref_std[0] = 1; ref_std = stdev(ref_std)
      standardized_stdev = 1. - std_normalized_counts/ref_std
      print ( " * Standardized stdev: %.2f" % standardized_stdev)
      self.set_balance(standardized_stdev)

    elif balance_method=="Hellinger":
      print(" * Checking balance with Hellinger-based entropy")

      values  = output_DF.unique()
      K = len(values)
     
      # calculate relative frequencies of all classes
      print(" * Value counts:")
      print(output_DF.value_counts())
      counts_classes = array(output_DF.value_counts())
      Pks = counts_classes/counts_classes.sum()
      #print(" * Detected classes proportions:", Pks)

      res = 0
      for k in range(K):
        res += sqrt(Pks[k]/K)
      numerator = 1-res
      denominator = 1-sqrt(1/K)
      result = 1-sqrt(numerator/denominator)
      print(" * Hellinger-based entropy: %.2f" % result)
      self.set_balance(result)


    else:
      raise Exception(" * %s data set balance method not supported, aborting." % balance_method)
   
    self._input_dataframe = input_DF
    self._output_dataframe = output_DF


  def set_balance(self, value):
    self._balance_value = value
    print (" * Balance set to: %.2f" % value)

  def set_numerosity(self, value):
    self._numerosity_value = value
    print (" * Numerosity ratio set to %.2f" % value)

  def set_outliers(self, value):
    self._outliers_ratio = value

  def set_compliance(self, compliance_dictionary={}):
    if all(type(v)==bool for v in compliance_dictionary.values()):
      self._compliance_value = len(list(filter(None, compliance_dictionary.values())))
    else:
      print ("ERROR: please specify a compliance dictionary.")
      exit()

  def set_quality(self, value):
    self._quality_value = value

  def set_incompleteness(self, value):
    self._incompleteness_value = value
    print(" * Incompleteness ratio set to %.2f" % value)

  def set_numsamples(self, value):
    self._numsamples = value

  def set_numfeatures(self, value):
    self._numfeatures = value

  def _set_maxsensitivity(self, value):
    self._maxsensitivity = value

  def calculate_fairness(self):
    if self._balance_value is None:
      raise Exception("ERROR: please calculate balance before assessing fairness of dataset")

    if self._numerosity_value is None:
      raise Exception("please calculate numerosity before assessing fairness of dataset")

    if self._outliers_ratio is None:
      raise Exception("please calculate outliers ratio before assessing fairness of dataset")

    if self._compliance_value is None:
      raise Exception("please calculate compliance before assessing fairness of dataset")

    if self._quality_value is None:
      raise Exception("please calculate quality before assessing fairness of dataset")

    if self._incompleteness_value is None:
      raise Exception("please calculate incompleteness before assessing fairness of dataset")

    self._reasoner.set_variable("balance", self._balance_value)
    self._reasoner.set_variable("numerosity", self._numerosity_value)
    self._reasoner.set_variable("unevenness", self._outliers_ratio)
    self._reasoner.set_variable("compliance", self._compliance_value)
    self._reasoner.set_variable("quality", self._quality_value) 
    self._reasoner.set_variable("incompleteness", self._incompleteness_value)
    self._reasoner.set_variable("correlation", self._maxsensitivity)

    res = self._reasoner.inference()

    return res["phi"]

  def produce_report(self, max_figures_per_row=4, 
    plot_fuzzy_sets=True, 
    plot_gauge=True, 
    model_file="",
    gauge_file=""
    ):
    """ Create a summary plot with all fuzzy sets, for all fairness metrics. """

    if plot_fuzzy_sets:
      self._reasoner.produce_figure(
                      outputfile=model_file,
                      max_figures_per_row=max_figures_per_row, 
                      element_dict={"quality": self._quality_value,
                                    "balance": self._balance_value,
                                    "numerosity": self._numerosity_value,
                                    "unevenness": self._outliers_ratio,
                                    "compliance": self._compliance_value,
                                    "incompleteness": self._incompleteness_value,
                                    "correlation": self._maxsensitivity })

      
    self._gauge = plt.figure(figsize=(15,7))
    ax = self._gauge.add_subplot(projection="polar")

    colors = ['#4dab6d', "#72c66e", "#c1da64", "#f6ee54", "#fabd57", "#f36d54"]
  
    values = arange(100,0,6)
    x_axis_vals = [0, pi*(1/6), pi*(2/6), pi*(3/6), pi*(4/6), pi*(5/6)]

    ax.bar(x=x_axis_vals, width=0.5, height=0.5, bottom=2,
       linewidth=3, edgecolor="white", color=colors, align="edge")

    plt.annotate("Excellent", xy=(0.16,2.1), rotation=-75, color="white", fontweight="bold");
    plt.annotate("Very bad", xy=(3.0,2.25), rotation=75, color="white", fontweight="bold");

    for loc, val in zip(x_axis_vals, values):
      plt.annotate(val, xy=(loc, 2.5), ha="right" if val<=20 else "left")

    # hide lines and ticks
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.spines['polar'].set_visible(False)

    value = self.calculate_fairness()

    plt.annotate("%.0f%%" % (value*100), xytext=(0,0), xy=((1-value)*pi, 2.0),
             arrowprops=dict(arrowstyle="wedge, tail_width=0.5", color="black", shrinkA=0),
             bbox=dict(boxstyle="circle", facecolor="black", linewidth=2.0, ),
             fontsize=30, color="white", ha="center"
            )

    # data set file name
    if self._dataset_file is not None:
      plt.text(0.5, 0.35, "Dataset: %s" % self._dataset_file, color="black", 
        ha="center", transform=ax.transAxes)

    # outlier detection method
    plt.text(0.5, 0.3, "Outlier detection method: %s" % self._used_outlier_method, color="black", ha="center",
      transform=ax.transAxes)

    # balance calculation method
    if self._used_balance_method=="Hellinger":
      balmethod = "Hellinger-based entropy"
    elif self._used_balance_method=="sigma-ratio":
      balmethod = "Ratio of sigmas (legacy)"
    plt.text(0.5, 0.25, "Balance calculation method: %s" % balmethod, color="black", ha="center",
      transform=ax.transAxes)

    # sensitive?
    if self._worst_correlation_name is not None:
      plt.text(0.5, 0.2, "Most sensitive variable: '%s', correlation with output: %.2f" % (self._worst_correlation_name, 
        self._maxsensitivity), color="black", ha="center", transform=ax.transAxes)


    self._gauge.tight_layout()

    if gauge_file!="": 
      self._gauge.savefig(gauge_file)
      print(" * Gauge exported to file", gauge_file)

    bar = "[%s%s]" % ( "█"*int(value*10), " "*(10-int(value*10)))
    print (" >> Data set's calculated fairness: %s %.1f%%" % (bar, value*100)) 
   

import seaborn as sns

sns.set_style("white")
sns.set_palette("husl", 3)

if __name__ == "__main__":

  pass