dataName <- "Saved_Data/100_patients/100_clean/qrisk_100.csv"
myData <- read.csv(dataName, check.names=FALSE)

test_all_rst <-  QRISK3_2017(data= myData, patid="patid", gender="gender",age="age",
                             atrial_fibrillation="AF", atypical_antipsy="Antipsychotic",
                             regular_steroid_tablets="Corticosteroid", erectile_disfunction="Erectile_dysfunction",
                             migraine="Migraine", rheumatoid_arthritis="Rheumatoid_arthritis",
                             chronic_kidney_disease="CKD345", severe_mental_illness="Severe_mental_illness",
                             systemic_lupus_erythematosis="SLE", blood_pressure_treatment="bp_treatment",
                             diabetes1="Diabetes_1", diabetes2="Diabetes_2", weight="weight", height="height",
                             ethiniciy="ethnicity_num", heart_attack_relative="family_history",
                             cholesterol_HDL_ratio="Total/HDL_ratio", systolic_blood_pressure="SBP",
                             std_systolic_blood_pressure="SBP_sd", smoke="smoking_num", townsend="townsend")
write.csv(test_all_rst, file='Saved_Data/100_patients/100_clean/qrisk_results.csv', row.names=FALSE, quote=FALSE)