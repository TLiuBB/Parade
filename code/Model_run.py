import os
from Qrisk_calculate import qrisk3
from Model_preprocess import preprocess
from Model_development import model_develop
from Final_plot import final_plot

base_path = '/Users/tonyliubb/not in iCloud/CPRD/Parade'

os.chdir(base_path)

"""
qrisk3(469496, base_path, prepare=False, calculate=False, plot=True, qrisk_sta=False, qrisk_sta_all=False,
       age_risk=False, plot_2=False, confirm=False)



preprocess(469496, prepare=True, region='London', bc_missing=True, imputation=True, bc_imputed=True,
           correlation=True)




model_develop(variable_set='qrisk', outcomes='cvd_all',
              LL_CoxPH=True, sk_CoxPH=True, sk_Rsf=False, sk_GBSM=False, xgbs=False, deeps=False, deeph=False,
              result=True)
"""
final_plot(finalauc=False, finalcalibration=False, external=None, variable_set='od', outcomes='cvd_all',
           external_plot=True)

