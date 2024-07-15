#import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

#user defined function
from extract_data_make_plots import rt_data_make_plots
from extract_data_make_plots import test_fun



#load data
patient_id = '19978'
patient_timepoint="acute"
#rt_data_make_plots(patient_id, patient_timepoint)
test_fun(patient_id, patient_timepoint)

patient_timepoint="fast"
#rt_data_make_plots(patient_id, patient_timepoint)

patient_id='19344'
patient_timepoint='acute'
#rt_data_make_plots(patient_id, patient_timepoint)

patient_timepoint='fast'
#rt_data_make_plots(patient_id, patient_timepoint)

patient_id='22725'
patient_timepoint='acute'
#rt_data_make_plots(patient_id, patient_timepoint)

patient_id='13990'
patient_timepoint='acute'
#rt_data_make_plots(patient_id, patient_timepoint)




 