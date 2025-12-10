import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import csv
import os

def eval_func(dataset,method):


    anomaly_rate = {
        'SMD': 0.5,#0.001, 
        'SMAP': 1, #2, 
        'MSL': 1,#0.1, 
        'SWaT': 0.1,# 0.005, 
        'UCR': 0.005, 
        'PSM':1
    }
    # anomaly_rate = {
    #     'SMD': 0.001, 
    #     'SMAP': 2, 
    #     'MSL': 0.1, 
    #     'SWaT': 0.005, 
    #     'UCR': 0.005, 
    #     'PSM':1,
    #     'NIPS_TS_Swan' : 0.9,
    #     'NIPS_TS_Water' : 1
    # }


    print("Dataset : ",dataset)
    #print("Model : ", model[b])
    print("Method : ", method)
    label = np.load('./fulldata/{}/plot/{}_labels.npy'.format(dataset, method))
    score = np.load('./fulldata/{}/plot/{}_loss.npy'.format(dataset, method))
    lossT = np.load('./fulldata/{}/plot/{}_lossT.npy'.format(dataset,method))

    score_df = pd.DataFrame(score)
    label_df = pd.DataFrame(label)

    attens_energy = lossT
    anormly_ratio =anomaly_rate[dataset]
    thresh = np.percentile(attens_energy, 100 - anormly_ratio)
    print("Threshold :", thresh)

    test_energy = score
    pred = (test_energy > thresh).astype(int)
    pred1 = pred.copy()
    gt = label

    pw_accuracy = accuracy_score(gt, pred)
    pw_precision, pw_recall, pw_f_score, pw_support = precision_recall_fscore_support(gt, pred, average='binary')
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(pw_accuracy, pw_precision, pw_recall, pw_f_score))

    beta = 0.6
    f_beta = precision_recall_balance_score(pw_precision, pw_recall, beta)
    print(f"Precision-Recall Balance Score: {f_beta:.4f}")

    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    pred2 = np.array(pred)
    gt = np.array(gt)

    pa_accuracy = accuracy_score(gt, pred)
    pa_precision, pa_recall, pa_f_score, pa_support = precision_recall_fscore_support(gt, pred, average='binary')
    print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(pa_accuracy, pa_precision, pa_recall, pa_f_score))


    pred_df = pd.DataFrame(pred1)
    pred_PA_df = pd.DataFrame(pred2)

    csv_data = {
    'method': [method],
    'threshold': [thresh],
    'F-beta': [f_beta],
    'PW-Accuracy': [pw_accuracy],
    'PW-Precision': [pw_precision],
    'PW-Recall': [pw_recall],
    'PW-F-score': [pw_f_score],
    'PA-Accuracy': [pa_accuracy],
    'PA-Precision': [pa_precision],
    'PA-Recall': [pa_recall],
    'PA-F-score': [pa_f_score],
    }

    df = pd.DataFrame(csv_data)

    csv_name = 'results_full/{}_additional.csv'.format(dataset)

    with open(csv_name, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=df.columns)
        if file.tell() == 0:  # 파일이 비어있는 경우 헤더를 작성
            writer.writeheader()
        writer.writerow(df.iloc[0].to_dict())

    # score_df.plot(figsize=(45,5))
    # plt.show()
    # label_df.plot(figsize=(45,5))
    # plt.show()
    # pred_df.plot(figsize=(45,5))
    # plt.show()
    # pred_PA_df.plot(figsize=(45,5))
    # plt.savefig('./fig/{}/{}_PA.png'.format(dataset,method))

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(45, 80))  # 각 그래프의 높이를 줄이고 전체 높이를 조정

    # 첫 번째 그래프 (score_df)
    score_df.plot(ax=axes[0], figsize=(45, 20), title='Score Plot')

    # 두 번째 그래프 (label_df)
    label_df.plot(ax=axes[1], figsize=(45, 20), title='Label Plot')

    # 세 번째 그래프 (pred_df)
    pred_df.plot(ax=axes[2], figsize=(45, 20), title='Prediction Plot (PW)')

    # 네 번째 그래프 (pred_PA_df)
    pred_PA_df.plot(ax=axes[3], figsize=(45, 20), title='Prediction Plot (PA)')

    # 레이아웃 조정 및 저장
    # plt.tight_layout()




    if not os.path.exists('./fig_Full/{}'.format(dataset)):
        os.makedirs('./fig_full/{}'.format(dataset))

    plt.savefig('./fig_Full/{}/{}.png'.format(dataset,method))
    plt.close()




def precision_recall_balance_score(precision, recall, beta=3):
    if precision + recall == 0:
        return 0.0
    
    beta_squared = beta ** 2
    prb_score = (1 + beta_squared) * (precision * recall) / (  beta_squared *precision +recall)
    return prb_score