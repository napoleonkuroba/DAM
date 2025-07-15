import numpy as np
import pandas as pd
import torch
from dam_model import DAM
from torch.nn import MSELoss
import time
from DataLoader import LinkDataSet,splitSet
from torch.utils.data import DataLoader
from layers.AutoEncoder import AutoEncoder


def DAM_Validation():
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size =32
    ep = 100
    mseMap={}
    device = "cuda"
    #
    model_types = ['full']
    for round in range(5):
        for model_type in model_types:
            for time_steps in [6]:
                for dataTitle in ["10"]:
                    mse = MSELoss()
                    links=LinkDataSet(flag='train', distance=dataTitle, step=time_steps,device='cpu')
                    loader = DataLoader(
                        links,
                        batch_size=batch_size,
                        num_workers=4,
                        pin_memory=True
                    )

                    #训练autoencoder
                    encoder=AutoEncoder().to(device)
                    encoder_optimizer = torch.optim.Adam(lr=0.001, params=encoder.parameters())
                    print("AutoEncoder Traning")
                    encoder_loss=0
                    for epoch in range(5):
                        for i,(x,_) in enumerate(loader):
                            encoder_optimizer.zero_grad()
                            x = x.to(device, non_blocking=True)
                            output=encoder(x)
                            loss=mse(x,output)
                            loss.backward()
                            encoder_loss=loss.item()
                            encoder_optimizer.step()
                        print("epoch:",epoch+1,"\t"," loss:",encoder_loss)
                    encoder.eval()

                    #训练dam
                    attention = DAM(model_type=model_type).to(device)
                    attention_optimizer = torch.optim.Adam(lr=0.001, params=attention.parameters())
                    epoch_loss = []
                    for epoch in range(ep):
                       for i, (x, y) in enumerate(loader):
                           x = x.to(device, non_blocking=True)
                           y = y.to(device, non_blocking=True)
                           input=encoder.encoder(x)
                           attention_optimizer.zero_grad()
                           output = attention(input)
                           output=encoder.decoder(output)
                           loss = mse(y, output)
                           loss.backward()
                           attention_optimizer.step()
                       print("epoch:", epoch)

                    #测试
                    mse = MSELoss()
                    with torch.no_grad():
                        testData = links.xTest.to(device, non_blocking=True)
                        testSet=encoder.encoder(testData)
                        attention_pre = attention(testSet)
                        attention_pre=encoder.decoder(attention_pre)

                    tt_1, tt_2, tt_3, ut_1, ut_2, ut_3, lms_1, lms_2, lms_3, mlr_1, mlr_2, mlr_3, cml_1, cml_2, cml_3 = splitSet(links.yTest,time_steps)
                    attention_tt_1, attention_tt_2, attention_tt_3, attention_ut_1, attention_ut_2, attention_ut_3, attention_lms_1, attention_lms_2, attention_lms_3, attention_mlr_1, attention_mlr_2, attention_mlr_3, attention_cml_1, attention_cml_2, attention_cml_3 = splitSet(
                        attention_pre,time_steps)
                    attentionarray = np.empty(0)
                    attentionarray = np.append(attentionarray, mse(tt_1, attention_tt_1).item())
                    attentionarray = np.append(attentionarray, mse(ut_1, attention_ut_1).item())
                    attentionarray = np.append(attentionarray, mse(lms_1, attention_lms_1).item())
                    attentionarray = np.append(attentionarray, mse(mlr_1, attention_mlr_1).item())
                    attentionarray = np.append(attentionarray, mse(cml_1, attention_cml_1).item())

                    attentionarray = np.append(attentionarray, mse(tt_2, attention_tt_2).item())
                    attentionarray = np.append(attentionarray, mse(ut_2, attention_ut_2).item())
                    attentionarray = np.append(attentionarray, mse(lms_2, attention_lms_2).item())
                    attentionarray = np.append(attentionarray, mse(mlr_2, attention_mlr_2).item())
                    attentionarray = np.append(attentionarray, mse(cml_2, attention_cml_2).item())

                    attentionarray = np.append(attentionarray, mse(tt_3, attention_tt_3).item())
                    attentionarray = np.append(attentionarray, mse(ut_3, attention_ut_3).item())
                    attentionarray = np.append(attentionarray, mse(lms_3, attention_lms_3).item())
                    attentionarray = np.append(attentionarray, mse(mlr_3, attention_mlr_3).item())
                    attentionarray = np.append(attentionarray, mse(cml_3, attention_cml_3).item())

                    attentionarray = attentionarray.reshape(-1, 5)
                    pd.DataFrame(attentionarray, columns=["TT", "UT", "LMS", "MLR", "CML"]).to_csv(
                        './result/Round-'+str(round)+ '-' + model_type + '-loss.csv',
                        index=False)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()  # 仅 Windows 需要
    DAM_Validation()