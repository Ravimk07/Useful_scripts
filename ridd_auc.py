from sklearn.metrics import roc_auc_score
import pandas as pd

pred_file1 = pd.read_csv('/home/aman/Desktop/aman/RIDD challenge/trial-6 k fold effb4 attn /RIDD_attention_fold3.csv')
pred_file2 = pd.read_csv('/home/aman/Desktop/aman/RIDD challenge/trial-6 k fold effb4 attn /RIDD_attention_fold3.csv')
pred_file3 = pd.read_csv('/home/aman/Desktop/aman/RIDD challenge/trial-6 k fold effb4 attn /RIDD_attention_fold3.csv')
pred_file4 = pd.read_csv('/home/aman/Desktop/aman/RIDD challenge/trial-6 k fold effb4 attn /RIDD_attention_fold3.csv')
pred_file5 = pd.read_csv('/home/aman/Desktop/aman/RIDD challenge/trial-6 k fold effb4 attn /RIDD_attention_fold3.csv')
true_file = pd.read_csv('/mnt/X/Ravi K/RIDD/Evaluation_Set/RFMiD_Validation_Labels.csv')

def find_auc_one_file(pred_file):
    score_1 = roc_auc_score(true_file['DR'],pred_file['DR'])
    score_2 = roc_auc_score(true_file['ARMD'],pred_file['ARMD'])
    score_3 = roc_auc_score(true_file['MH'],pred_file['MH'])
    score_4 = roc_auc_score(true_file['DN'],pred_file['DN'])
    score_5 = roc_auc_score(true_file['MYA'],pred_file['MYA'])
    score_6 = roc_auc_score(true_file['BRVO'],pred_file['BRVO'])
    score_7 = roc_auc_score(true_file['TSLN'],pred_file['TSLN'])
    score_8 = roc_auc_score(true_file['ERM'],pred_file['ERM'])
    score_9 = roc_auc_score(true_file['LS'],pred_file['LS'])
    score_10 = roc_auc_score(true_file['MS'],pred_file['MS'])
    score_11 = roc_auc_score(true_file['CSR'],pred_file['CSR'])
    score_12 = roc_auc_score(true_file['ODC'],pred_file['ODC'])
    score_13 = roc_auc_score(true_file['CRVO'],pred_file['CRVO'])
    score_14 = roc_auc_score(true_file['TV'],pred_file['TV'])
    score_15 = roc_auc_score(true_file['AH'],pred_file['AH'])
    score_16 = roc_auc_score(true_file['ODP'],pred_file['ODP'])
    score_17 = roc_auc_score(true_file['ODE'],pred_file['ODE'])
    score_18 = roc_auc_score(true_file['ST'],pred_file['ST'])
    score_19 = roc_auc_score(true_file['AION'],pred_file['AION'])
    score_20 = roc_auc_score(true_file['PT'],pred_file['PT'])
    score_21 = roc_auc_score(true_file['RT'],pred_file['RT'])
    score_22 = roc_auc_score(true_file['RS'],pred_file['RS'])
    score_23 = roc_auc_score(true_file['CRS'],pred_file['CRS'])
    score_24 = roc_auc_score(true_file['EDN'],pred_file['EDN'])
    score_25 = roc_auc_score(true_file['RPEC'],pred_file['RPEC'])
    score_26 = roc_auc_score(true_file['MHL'],pred_file['MHL'])
    score_27 = roc_auc_score(true_file['RP'],pred_file['RP'])
    score_28 = roc_auc_score(true_file['OTHER'],pred_file['OTHER'])
    
    
    total = score_1 + score_2 + score_3 + score_4 + score_5 + score_6 + score_7 + score_8 + score_9 + score_10 + score_11 + score_12 + score_13 + score_14 + score_15 + score_16 + score_17 + score_18 + score_19 +  score_20 +  score_21+  score_22+  score_23+  score_24+  score_25+  score_26+  score_27+  score_28 
    avg = total/28.
    return avg

def find_auc_five_file():
    k1=0.2
    k2=0.2
    k3=0.2
    k4=0.2
    k5=0.5
    score_1 = roc_auc_score(true_file['DR'],k1*pred_file1['DR']+k2*pred_file2['DR']+k3*pred_file3['DR']+k4*pred_file4['DR']+k5*pred_file5['DR'])
    score_2 = roc_auc_score(true_file['ARMD'],k1*pred_file1['ARMD']+k2*pred_file2['ARMD']+k3*pred_file3['ARMD']+k4*pred_file4['ARMD']+k5*pred_file5['ARMD'])
    score_3 = roc_auc_score(true_file['MH'],k1*pred_file1['MH']+k2*pred_file2['MH']+k3*pred_file3['MH']+k4*pred_file4['MH']+k5*pred_file5['MH'])
    score_4 = roc_auc_score(true_file['DN'],k1*pred_file1['DN']+k2*pred_file2['DN']+k3*pred_file3['DN']+k4*pred_file4['DN']+k5*pred_file5['DN'])
    score_5 = roc_auc_score(true_file['MYA'],k1*pred_file1['MYA']+k2*pred_file2['MYA']+k3*pred_file3['MYA']+k4*pred_file4['MYA']+k5*pred_file5['MYA'])
    score_6 = roc_auc_score(true_file['BRVO'],k1*pred_file1['BRVO']+k2*pred_file2['BRVO']+k3*pred_file3['BRVO']+k4*pred_file4['BRVO']+k5*pred_file5['BRVO'])
    score_7 = roc_auc_score(true_file['TSLN'],k1*pred_file1['TSLN']+k2*pred_file2['TSLN']+k3*pred_file3['TSLN']+k4*pred_file4['TSLN']+k5*pred_file5['TSLN'])
    score_8 = roc_auc_score(true_file['ERM'],k1*pred_file1['ERM']+k2*pred_file2['ERM']+k3*pred_file3['ERM']+k4*pred_file4['ERM']+k5*pred_file5['ERM'])
    score_9 = roc_auc_score(true_file['LS'],k1*pred_file1['LS']+k2*pred_file2['LS']+k3*pred_file3['LS']+k4*pred_file4['LS']+k5*pred_file5['LS'])
    score_10 = roc_auc_score(true_file['MS'],k1*pred_file1['MS']+k2*pred_file2['MS']+k3*pred_file3['MS']+k4*pred_file4['MS']+k5*pred_file5['MS'])
    score_11 = roc_auc_score(true_file['CSR'],k1*pred_file1['CSR']+k2*pred_file2['CSR']+k3*pred_file3['CSR']+k4*pred_file4['CSR']+k5*pred_file5['CSR'])
    score_12 = roc_auc_score(true_file['ODC'],k1*pred_file1['ODC']+k2*pred_file2['ODC']+k3*pred_file3['ODC']+k4*pred_file4['ODC']+k5*pred_file5['ODC'])
    score_13 = roc_auc_score(true_file['CRVO'],k1*pred_file1['CRVO']+k2*pred_file2['CRVO']+k3*pred_file3['CRVO']+k4*pred_file4['CRVO']+k5*pred_file5['CRVO'])
    score_14 = roc_auc_score(true_file['TV'],k1*pred_file1['TV']+k2*pred_file2['TV']+k3*pred_file3['TV']+k4*pred_file4['TV']+k5*pred_file5['TV'])
    score_15 = roc_auc_score(true_file['AH'],k1*pred_file1['AH']+k2*pred_file2['AH']+k3*pred_file3['AH']+k4*pred_file4['AH']+k5*pred_file5['AH'])
    score_16 = roc_auc_score(true_file['ODP'],k1*pred_file1['ODP']+k2*pred_file2['ODP']+k3*pred_file3['ODP']+k4*pred_file4['ODP']+k5*pred_file5['ODP'])
    score_17 = roc_auc_score(true_file['ODE'],k1*pred_file1['ODE']+k2*pred_file2['ODE']+k3*pred_file3['ODE']+k4*pred_file4['ODE']+k5*pred_file5['ODE'])
    score_18 = roc_auc_score(true_file['ST'],k1*pred_file1['ST']+k2*pred_file2['ST']+k3*pred_file3['ST']+k4*pred_file4['ST']+k5*pred_file5['ST'])
    score_19 = roc_auc_score(true_file['AION'],k1*pred_file1['AION']+k2*pred_file2['AION']+k3*pred_file3['AION']+k4*pred_file4['AION']+k5*pred_file5['AION'])
    score_20 = roc_auc_score(true_file['PT'],k1*pred_file1['PT']+k2*pred_file2['PT']+k3*pred_file3['PT']+k4*pred_file4['PT']+k5*pred_file5['PT'])
    score_21 = roc_auc_score(true_file['RT'],k1*pred_file1['RT']+k2*pred_file2['RT']+k3*pred_file3['RT']+k4*pred_file4['RT']+k5*pred_file5['RT'])
    score_22 = roc_auc_score(true_file['RS'],k1*pred_file1['RS']+k2*pred_file2['RS']+k3*pred_file3['RS']+k4*pred_file4['RS']+k5*pred_file5['RS'])
    score_23 = roc_auc_score(true_file['CRS'],k1*pred_file1['CRS']+k2*pred_file2['CRS']+k3*pred_file3['CRS']+k4*pred_file4['CRS']+k5*pred_file5['CRS'])
    score_24 = roc_auc_score(true_file['EDN'],k1*pred_file1['EDN']+k2*pred_file2['EDN']+k3*pred_file3['EDN']+k4*pred_file4['EDN']+k5*pred_file5['EDN'])
    score_25 = roc_auc_score(true_file['RPEC'],k1*pred_file1['RPEC']+k2*pred_file2['RPEC']+k3*pred_file3['RPEC']+k4*pred_file4['RPEC']+k5*pred_file5['RPEC'])
    score_26 = roc_auc_score(true_file['MHL'],k1*pred_file1['MHL']+k2*pred_file2['MHL']+k3*pred_file3['MHL']+k4*pred_file4['MHL']+k5*pred_file5['MHL'])
    score_27 = roc_auc_score(true_file['RP'],k1*pred_file1['RP']+k2*pred_file2['RP']+k3*pred_file3['RP']+k4*pred_file4['RP']+k5*pred_file5['RP'])
    score_28 = roc_auc_score(true_file['OTHER'],k1*pred_file1['OTHER']+k2*pred_file2['OTHER']+k3*pred_file3['OTHER']+k4*pred_file4['OTHER']+k5*pred_file5['OTHER'])
    
    
    total = score_1 + score_2 + score_3 + score_4 + score_5 + score_6 + score_7 + score_8 + score_9 + score_10 + score_11 + score_12 + score_13 + score_14 + score_15 + score_16 + score_17 + score_18 + score_19 +  score_20 +  score_21+  score_22+  score_23+  score_24+  score_25+  score_26+  score_27+  score_28 
    avg = total/28.
    return avg

print('auc fold 1: '+ str(find_auc_one_file(pred_file1)))
print('auc fold 2: '+ str(find_auc_one_file(pred_file2)))
print('auc fold 3: '+ str(find_auc_one_file(pred_file3)))
print('auc fold 4: '+ str(find_auc_one_file(pred_file4)))
print('auc fold 5: '+ str(find_auc_one_file(pred_file5)))
print('auc avg:' + str(find_auc_five_file()))




