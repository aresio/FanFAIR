# FanFAIR
Semi-automatic assessment of datasets fairness

## What is FanFAIR
FanFAIR is a rule-based approach based on fuzzy logic able to calculate some fairness metrics over a dataset and combine them into a single score, enabling a semi-automatic evaluation of a dataset in algorithmic fairness research.

## Using FanFAIR
FanFAIR is designed to be as automatic as possible. However, two metrics (quality, compliance) require human intervention. Here is an example of analysis performed with FanFAIR:

```
from fanfair import FanFAIR

FF = FanFAIR(dataset="myfile.csv", target_column="output")
FF.set_compliance( {"data_protection_law": True,
                    "copyright_law": True,
                    "medical_law": True,
                    "non_discrimination_law": False,
                    "ethics": False})
FF.set_quality(0.9)
FF.produce_report()
```

The analysis is automatically performed by calling the ```produce_report``` method, which generates two main figures: the gauge with the overall fairness score (from 0% to 100%), and the plots of the linguistic variables of the fuzzy model, which provide a summary of the metrics for the dataset's fairenss features.


## Citing FanFAIR 
If you find FanFAIR useful for your research, please cite our project as follows:

> Gallese C., Scantamburlo T., Manzoni L., Nobile M.S.: Investigating Semi-Automatic Assessment of Data Sets Fairness by Means of Fuzzy Logic, Proceedings of the 20th IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology (IEEE CIBCB 2023), 2023 

If you need additional information, or want to see additional metrics implemented in FanFAIR, please feel free to contact Dr. Chiara Gallese (chiara.gallese@unito.it). 

## Acknowledgements
![FanFAIR is funded by the European Union](assets/EN_FundedbytheEU_RGB_POS.png)
