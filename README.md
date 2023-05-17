# Nowcasting with signatures

## Overview

In this repository, we include the code to generate and reproduce the results
in our paper Nowcasting with signature methods.

We have also generalise the code used here and created a Python package
[SigNow](https://github.com/datasciencecampus/SigNow_ONS_Turing) which 
facilitates the implementation of the signature method for other economic 
nowcasting applications of your choice.

## Set-up the environment
Install the requirements through (or using your preferred package 
management/virtualenv set-up)
```
pip install -r requirements.txt
```

To run the experiments for the paper, activate the python virtual environment 
with the appropriate pre-requisites installed and see the subsections below.
    
The data is stored in `data` and the results will be saved to `results`.

## Simulation experiments

To run the experiments with the default configs, simply use

```
python src/simulation_main.py
```

If you are using the default config file, then add this as an argument
```
python src/simulation_main.py <your_config_file>.yaml
```

For the appendix in our paper we discussed comparison of the coefficients
against the theoretical ones. This can be seen in the notebook 
`sig_coefficients.ipynb` in the folder `notebooks`.


## US GDP Growth

The NY Fed staff nowcast is found on their 
[website](https://www.newyorkfed.org/research/policy/nowcast).

The indicator variables are taken from [FRED](https://fred.stlouisfed.org/).

To run the experiments with the default configs and hyperparameters to search 
over
```
python src/usgdp_main.py
```
Again, modified configurations/hyperparameters can be used
```
python src/usgdp_main.py <your_config_file>.yaml
```

Please note this can take about 1 day to run.

Once this is completed, the plots can be generated with the notebook 
`usgdp_results_final.ipynb` in the folder `notebooks`.

## Fuel analysis

The target data available comes from the 
[Weekly road fuel prices](https://www.gov.uk/government/statistics/weekly-road-fuel-prices) 
series published by the Department for Business, Energy \& Industrial Strategy
(BEIS).

To nowcast this series, we use the daily close price of WisdomTree Brent Crude 
Oil (BRNT), an ETC, at the London Stock Exchange. The data (in USD) is publicly
available on [Yahoo](https://finance.yahoo.com/quote/BRNT.L/history?p=BRNT.L), 
and we then use a [daily USD-GBP exchange rate](https://uk.investing.com/currencies/usd-gbp-historical-data) 
to convert the price of BRNT to GBP.

To run the experiments with the default configs and hyperparameters to search 
over
```
python src/fuel_analysis.py
```
or with your set of configs 
```
python src/fuel_analysis.py <your_config_file>.yaml
```

## Citation

```bibtex
@article{cohen2023nowcasting,
  title={Nowcasting with signature methods},
  author={Cohen, Samuel N. and Lui, Silvia and Malpass, Will and Mantoan, 
  Giulia and Nesheim, Lars and de Paula, \'{A}ureo and Reeves, Andrew and
  Scott, Craig and Small, Emma and Yang, Lingyi},
  journal={arXiv},
  year={2023}
}