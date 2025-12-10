import pickle
import os
import pandas as pd
from tqdm import tqdm
import torch
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
import csv
from utils_eval import *
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def convert_to_windows(data, w_size):
    windows = [] 
    for i, g in enumerate(data): 
        if i > w_size: 
            w = data[i-w_size+1:i+1]
        elif i == w_size:
            w = data[1:w_size+1]
        else: 
            w = torch.cat([data[0].repeat(w_size-i, 1), data[1:i+1]])
        windows.append(w)
    return torch.stack(windows)

def convert_to_windows_gap(data, w_size):
    windows = []
    for i in tqdm(range(len(data) - w_size)):  # 마지막 윈도우는 따로 처리
        # 시간 간격을 계산하여 10초 * w_size 조건을 확인
        if (pd.to_datetime(data.index[i + w_size]) - pd.to_datetime(data.index[i])) / pd.Timedelta(seconds=1) == 10 * w_size:
            w = data[i:i + w_size]
            windows.append(w)
    
    # 마지막 윈도우 처리 (마지막에서 w_size만큼 떨어진 구간)
    if (pd.to_datetime(data.index[-1]) - pd.to_datetime(data.index[-w_size])) / pd.Timedelta(seconds=1) == 10 * w_size:
        windows.append(data[-w_size:])

    # 윈도우를 텐서로 변환하여 반환
    return torch.stack([torch.tensor(window.values) for window in windows])


#원본 load_dataset
def load_dataset(dataset):
	loader = []
	for file in ['train', 'test_tra', 'label4']:
		if file == 'train':
			data = np.load('data/{}/리뉴얼/{}.npy'.format(args.dataset, file))
			loader.append(data)
		elif file == 'test':
			data = np.load('data/{}/리뉴얼/{}.npy'.format(args.dataset,file))
			data = data[:int((data.shape[0]))]
			loader.append(data)
		else:
			data = np.load('data/{}/리뉴얼/{}.npy'.format(args.dataset,file))
			data = data[:int((data.shape[0]))]
			loader.append(data)


	if args.less: loader[0] = cut_array(0.2, loader[0])
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	return train_loader, test_loader, labels


# 정상 데이터 테스트할때쓰는 load dataset
# def load_dataset(dataset, list):
# 	loader = []
# 	for file in list:
# 		if 'train' in file:
# 			data = np.load('data/{}/리뉴얼/march/{}.npy'.format(args.dataset, file))
# 			loader.append(data)
# 		elif 'normal' in file:
# 			data = np.load('data/{}/리뉴얼/정상데이터/{}.npy'.format(args.dataset,file))
# 			data = data[:int((data.shape[0]))]
# 			loader.append(data)
# 		else:
# 			data = np.load('data/{}/리뉴얼/정상데이터/{}.npy'.format(args.dataset,file))
# 			data = data[:int((data.shape[0]))]
# 			loader.append(data)


# 	if args.less: loader[0] = cut_array(0.2, loader[0])
# 	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
# 	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
# 	labels = loader[2]
# 	return train_loader, test_loader, labels


def save_model(model, optimizer, scheduler, epoch, accuracy_list):
    folder = f'checkpoints/{args.model}_{args.dataset}_renew_d64/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims,args):
    import src.models
    model_class = getattr(src.models, modelname)
    model = model_class(dims).double().to(device)  # 모델을 GPU로 이동
    model.batch = args.batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'checkpoints/{args.model}_{args.dataset}_renew_d64/model.ckpt'
    if os.path.exists(fname) and (not args.retrain or args.test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1; accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list



def plot_attention(attention_weights, layer_idx=0, head_idx=0):
    """
    시각화를 위한 Attention Weights 플로팅 함수
    :param attention_weights: 모델의 Attention Weights 리스트
    :param layer_idx: 사용할 Transformer Layer의 인덱스
    :param head_idx: 사용할 Attention Head의 인덱스
    """
    attn = attention_weights[layer_idx].detach().cpu().numpy()  # 선택한 레이어의 attention
    print(f"Attention shape before adjustment: {attn[head_idx].shape}")

    # 차원이 (60, 1)인 경우 (60,)으로 변환
    if attn[head_idx].shape[-1] == 1:
        data_to_plot = attn[head_idx].squeeze(-1)
    # 차원이 (60, 1)인 경우 (60, 60)으로 확장
    elif attn[head_idx].shape[0] == attn[head_idx].shape[-1]:
        data_to_plot = attn[head_idx]
    else:
        raise ValueError(f"Unsupported attention shape: {attn[head_idx].shape}")

    sns.heatmap(data_to_plot, cmap='viridis')
    plt.title(f'Attention Map - Layer {layer_idx}, Head {head_idx}')
    plt.xlabel('Key')
    plt.ylabel('Query')

    plt.savefig('attentionmap/attentionmap.png')
    plt.show()


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
    l = nn.MSELoss(reduction='mean' if training else 'none').to(device)
    feats = dataO.shape[1]

    if 'TranAD' in model.name:
        l = nn.MSELoss(reduction='none').to(device)
        data_x = data 
        
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch #if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1; w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d, _ in tqdm(dataloader):
                local_bs = d.shape[0]
                d = d.to(device)
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem, save_attention=False)
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
                if isinstance(z, tuple): z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model = model.cpu()
            loss_list = torch.tensor([], dtype=torch.float32)#.to(device)
            for d, _ in tqdm(dataloader):
                d = d#.to(device)
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem, save_attention=True)
                loss = (1/2) * l(z[0], elem)[0] + (1/2) * l(z[1], elem)[0]  # loss=[128,25]
                #print("loss.shape", loss.shape)
                #loss = torch.mean(loss, axis=1)  # loss=[128]
                loss_list = torch.cat((loss_list, loss.detach().cpu()), dim=0)
            return (loss_list).numpy(), 1,1
        
    elif 'Transformer' in model.name:
        l = nn.MSELoss(reduction='none').to(device)
        data_x = data 
        
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch #if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1; 
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for window, _ in tqdm(dataloader):
                local_bs = window.shape[0]
                window = window.to(device)
                #print("d.shape", d.shape )
                #window = d.permute(1, 0, 2)  #[60, 64, 15]
                #print("window.shape", window.shape )
                z = model(window,save_attention=False)
                l1 = l(z, window) #if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
                l1s.append(torch.mean(l1).item())
                #print("torch.mean(l1)", torch.mean(l1).shape)
                loss = torch.mean(l1)
                #print("loss.shape", loss.shape)
                optimizer.zero_grad()
                #print("loss.shape", loss.shape)
                loss.backward(retain_graph=True)
                optimizer.step()
                l1s.append(torch.mean(l1).item())
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            # CPU버전
            # model = model.cpu()
            # loss_list = torch.tensor([], dtype=torch.float32)
            # for window, _ in tqdm(dataloader):
            #     window = window
            #     local_bs = window.shape[0]
            #     #window = d.permute(1, 0, 2)     
            #     z = model(window)
            #     loss = l(z, window) # loss=[128,25]
            #     #print("loss.shape", loss.shape)
            #     #loss = torch.mean(loss, axis=1)  # loss=[128]
            #     #print("loss.shape", loss.shape)
            #     loss_list = torch.cat((loss_list, loss.detach()), dim=0)
            #     #print("loss_list.shape", loss_list.shape)
            # return (loss_list).numpy(), 1
        
            loss_list = torch.tensor([], dtype=torch.float32)
            z_list = torch.tensor([], dtype=torch.float32)
            input_list = torch.tensor([], dtype=torch.float32)
            model = model.cpu()
            for window, _ in tqdm(dataloader):
                window = window  # CPU에서 데이터 유지
                local_bs = window.shape[0]

                # 모델 추론
                z = model(window, save_attention=False)  # GPU 코드에서 save_attention 관련 부분 제거

                # 손실 계산
                loss = l(z, window)  # loss=[batch_size, sequence_length]

                # 텐서를 concat
                loss_list = torch.cat((loss_list, loss.detach()), dim=0)
                z_list = torch.cat((z_list, z.detach()), dim=0)
                input_list = torch.cat((input_list, window.detach()), dim=0)

            return loss_list.numpy(), z_list.numpy(), input_list.numpy()
        
        #GPU버전
            # loss_list = torch.tensor([], dtype=torch.float32).to(device)
            # z_list = torch.tensor([], dtype=torch.float32).to(device)
            # input_list = torch.tensor([], dtype=torch.float32).to(device)
            # for window, _ in tqdm(dataloader):
            #     window = window.to(device)
            #     local_bs = window.shape[0]
            #     #window = d.permute(1, 0, 2)     
            #     z = model(window,save_attention=True)
            #     #print("z.shape", z.shape)
            #     loss = l(z, window) # loss=[128,25]
            #     #print("loss.shape", loss.shape)
            #     #loss = torch.mean(loss, axis=1)  # loss=[128]
            #     #print("loss.shape", loss.shape)
            #     loss_list = torch.cat((loss_list, loss.detach()), dim=0)
            #     z_list = torch.cat((z_list, z.detach()), dim=0)
            #     input_list  = torch.cat((input_list, window.detach()), dim=0)
            #     #print("loss_list.shape", loss_list.shape)
            # return (loss_list).cpu().numpy(), z_list.cpu().numpy(), input_list.cpu().numpy()

        
####################################################################################################################################################################################################################################


# # ### Main script
# if __name__ == '__main__':
#     print(args)
#     train_loader, test_loader, labels = load_dataset(args.dataset)
#     model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1], args)

#     #Prepare data
#     # trainD, testD = next(iter(train_loader)), next(iter(test_loader))
#     # print(trainD.shape)
#     # print(testD.shape)
#     # trainO, testO = trainD, testD  
#     # if 'TranAD' or 'Transformer' in model.name: 
#     #     trainD, testD = convert_to_windows(trainD, model.n_window), convert_to_windows(testD, model.n_window)
#     #     print(trainD.shape)  # torch.Size([1600, 10])
#     #     print(testD.shape)  # torch.Size([5900, 10])
    
#     with open("data/MetroPT/고장 4번/columns.pkl", "rb") as f:
#         list_ex = pickle.load(f)

#     trainD, testD = next(iter(train_loader)), next(iter(test_loader))
#     trainO, testO = trainD, testD  
#     df_new = pd.read_parquet('data/MetroPT/고장 4번/df_new.parquet')
#     df_new.dropna(how='any', inplace = True)

#     rawindex = df_new.index 
#     a = df_new[:'2020-03-02 00:00:00']
#     #b = df_new['2020-08-06 00:00:00':'2020-09-01 00:00:00']
#     trainindex = a.index
#     #testindex = rawindex[551734:570423]

#     trainD = pd.DataFrame(trainD, columns = list_ex)
#     trainD.index = trainindex
#     #testD = pd.DataFrame(testD, columns = list_ex)
#     #testD.index = testindex
#     if 'TranAD' or 'Transformer' in model.name: 
#         trainD, testD = convert_to_windows_gap(trainD, model.n_window), convert_to_windows(testD, model.n_window)
#         print(trainD.shape)  # torch.Size([1600, 10])
#         print(testD.shape)  # torch.Size([5900, 10])

#     ### Training phase
#     if not args.test:
#         print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
#         num_epochs = args.total_epochs; e = epoch + 1; start = time()
#         for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
#             lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
#             accuracy_list.append((lossT, lr))
#         print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
#         save_model(model, optimizer, scheduler, e, accuracy_list)
#         plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

#     ### Testing phase
#     model.eval()

#     print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
#     loss, y_pred, input = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

#     df = pd.DataFrame()
#     labelsFinal = (torch.sum(torch.tensor(labels), axis=1) >= 1).int()

#     np.save('data/MetroPT/리뉴얼/march/plot/Transformer_many_layer/loss4.npy', loss)
#     np.save('data/MetroPT/리뉴얼/march/plot/Transformer_many_layer/pred4.npy', y_pred)
#     np.save('data/MetroPT/리뉴얼/march/plot/Transformer_many_layer/input4.npy', input)
#     print("loss.shape" ,loss.shape)
#     print("y_pred.shape" ,y_pred.shape)
#     print("input.shape" ,input.shape)





####################################################################################################################################################################################################################################
#Visualize Attention
### Main script
# if __name__ == '__main__':
#     print(args)
#     train_loader, test_loader, labels = load_dataset(args.dataset)
#     model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1], args)

#     #Prepare data
#     trainD, testD = next(iter(train_loader)), next(iter(test_loader))
#     print(trainD.shape)
#     print(testD.shape)
#     trainO, testO = trainD, testD  
#     if 'TranAD' or 'Transformer' in model.name: 
#         trainD, testD = convert_to_windows(trainD, model.n_window), convert_to_windows(testD, model.n_window)
#         print(trainD.shape)  # torch.Size([1600, 10])
#         print(testD.shape)  # torch.Size([5900, 10])
    

# 파일 경로를 읽기 모드로 열어서 데이터 불러오기
    # with open("data/MetroPT/고장 4번/columns.pkl", "rb") as f:
    #     list_ex = pickle.load(f)

    # trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    # trainO, testO = trainD, testD  
    # rawindex = pd.read_parquet('data/MetroPT/고장 4번/df_new.parquet').index 
    # trainindex = rawindex[:450000]
    # testindex = rawindex[500000:600000]
    # trainD = pd.DataFrame(trainD, columns = list_ex)
    # trainD.index = trainindex
    # testD = pd.DataFrame(testD, columns = list_ex)
    # testD.index = testindex
    # if 'TranAD' or 'Transformer' in model.name: 
    #     trainD, testD = convert_to_windows_gap(trainD, model.n_window), convert_to_windows_gap(testD, model.n_window)
    #     print(trainD.shape)  # torch.Size([1600, 10])
    #     print(testD.shape)  # torch.Size([5900, 10])

         
         

# # Visualize Attention


#     ### Training phase
#     if not args.test:
#         print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
#         num_epochs = args.total_epochs; e = epoch + 1; start = time()
#         for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
#             lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
#             accuracy_list.append((lossT, lr))
#         print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
#         save_model(model, optimizer, scheduler, e, accuracy_list)
#         plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

#     ### Testing phase
#     model.eval()
#     model.encoder_attention_weights = []  # Clear stored attention weights

#     print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
#     loss, y_pred, input = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

#     with open("attentionmap/네번째고장/selfattentionmap_encoder.pkl","wb") as f:
#         pickle.dump(model.encoder_attention_weights, f)
#     with open("attentionmap/네번째고장/crossattentionmap.pkl","wb") as f:
#         pickle.dump(model.decoder_cross_attention_weights, f)
#     with open("attentionmap/네번째고장/selfattentionmap_decoder.pkl","wb") as f:
#         pickle.dump(model.decoder_self_attention_weights, f)


#     df = pd.DataFrame()
#     #lossT, _ ,_= backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
#     #lossTfinal, lossFinal = lossT, loss
#     lossFinal = loss
#     labelsFinal = (torch.sum(torch.tensor(labels), axis=1) >= 1).int()

#     #np.save('data/{}/정상데이터/plot/{}/loss.npy'.format(args.dataset,  args.model), lossFinal)
#     #np.save('data/{}/{}/lossT1.npy'.format(args.dataset,  args.model), lossT)
#     #np.save('data/{}/{}/data split/win120_d128/labels1.npy'.format(args.dataset,  args.model), labelsFinal.cpu().numpy())
#     #np.save('data/{}/정상데이터/plot/{}/pred.npy'.format(args.dataset,  args.model), y_pred)
#     #np.save('data/{}/정상데이터/plot/{}/input.npy'.format(args.dataset,  args.model), input)
#     print("score.shape", lossFinal.shape)
#     print("label.shape", labelsFinal.shape)

####################################################################################################################################################################################################################################
 
# if __name__ == '__main__':
#     print(args)
#     list_train_normal_label = []
#     for i in range(1,36):
#         list_train_normal_label = ['train', 'test{}'.format(i), 'label{}'.format(i)]
#         #list_train_normal_label.append(b)

#         train_loader, test_loader, labels = load_dataset(args.dataset, list_train_normal_label)
#         model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1], args)

#         #Prepare data
#         trainD, testD = next(iter(train_loader)), next(iter(test_loader))
#         print(trainD.shape)
#         print(testD.shape)
#         trainO, testO = trainD, testD  
#         if 'TranAD' or 'Transformer' in model.name: 
#             trainD, testD = convert_to_windows(trainD, model.n_window), convert_to_windows(testD, model.n_window)
#             print(trainD.shape)  # torch.Size([1600, 10])
#             print(testD.shape)  # torch.Size([5900, 10])
            

#         ### Training phase
#         if not args.test:
#             print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
#             num_epochs = args.total_epochs; e = epoch + 1; start = time()
#             for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
#                 lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
#                 accuracy_list.append((lossT, lr))
#             print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
#             save_model(model, optimizer, scheduler, e, accuracy_list)
#             plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

#         ### Testing phase
#         model.eval()
#         print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
#         loss, y_pred, input = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
#         df = pd.DataFrame()
#         #lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
#         #lossTfinal, lossFinal = lossT, loss
#         lossFinal = loss
#         labelsFinal = (torch.sum(torch.tensor(labels), axis=1) >= 1).int()



#         np.save('data/MetroPT/리뉴얼/정상데이터/plot/Transformer_layer2/loss{}.npy'.format(i), loss)
#         np.save('data/MetroPT/리뉴얼/정상데이터/plot/Transformer_layer2/pred{}.npy'.format(i), y_pred)
#         np.save('data/MetroPT/리뉴얼/정상데이터/plot/Transformer_layer2/input{}.npy'.format(i), input)
#         print("loss.shape" ,loss.shape)
#         print("y_pred.shape" ,y_pred.shape)
#         print("input.shape" ,input.shape)



        # np.save('data/{}/정상데이터/plot2/{}/loss{}.npy'.format(args.dataset,  args.model, i), lossFinal)
        # #np.save('data/{}/{}/lossT1.npy'.format(args.dataset,  args.model), lossTfinal)
        # #np.save('data/{}/{}/data split/win120_d128/labels1.npy'.format(args.dataset,  args.model), labelsFinal.cpu().numpy())
        # np.save('data/{}/정상데이터/plot2/{}/pred{}.npy'.format(args.dataset,  args.model,i), y_pred)
        # np.save('data/{}/정상데이터/plot2/{}/input{}.npy'.format(args.dataset,  args.model,i), input)
        # print("score.shape", lossFinal.shape)
        # print("label.shape", labelsFinal.shape)





####################################################################################################################################################################################################################################
# 이 코드는 학습데이터를 다시 테스트하는 코드
# ### Main script
if __name__ == '__main__':
    print(args)
    train_loader, test_loader, labels = load_dataset(args.dataset)
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1], args)

    #Prepare data
    # trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    # print(trainD.shape)
    # print(testD.shape)
    # trainO, testO = trainD, testD  
    # if 'TranAD' or 'Transformer' in model.name: 
    #     trainD, testD = convert_to_windows(trainD, model.n_window), convert_to_windows(testD, model.n_window)
    #     print(trainD.shape)  # torch.Size([1600, 10])
    #     print(testD.shape)  # torch.Size([5900, 10])
    
    with open("data/MetroPT/고장 4번/columns.pkl", "rb") as f:
        list_ex = pickle.load(f)

    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD  
    df_new = pd.read_parquet('data/MetroPT/고장 4번/df_new.parquet')
    df_new.dropna(how='any', inplace = True)

    rawindex = df_new.index 
    a = df_new[:'2020-03-02 00:00:00']
    b = df_new['2020-08-06 00:00:00':'2020-09-01 00:00:00']
    traindata = pd.concat([a,b], axis =0)
    trainindex = traindata.index
    #testindex = traindata.index

    trainD = pd.DataFrame(trainD, columns = list_ex)
    trainD.index = trainindex
    #testD = pd.DataFrame(testD, columns = list_ex)
    #testD.index = testindex
    if 'TranAD' or 'Transformer' in model.name: 
        trainD, testD = convert_to_windows_gap(trainD, model.n_window), convert_to_windows(testD, model.n_window)
        print(trainD.shape)  # torch.Size([1600, 10])
        print(testD.shape)  # torch.Size([5900, 10])

    ### Training phase
    if not args.test:
        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        num_epochs = args.total_epochs; e = epoch + 1; start = time()
        for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
            lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
            accuracy_list.append((lossT, lr))
        print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
        save_model(model, optimizer, scheduler, e, accuracy_list)
        plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

    ### Testing phase
    model.eval()

    print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
    loss, y_pred, input = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

    df = pd.DataFrame()
    labelsFinal = (torch.sum(torch.tensor(labels), axis=1) >= 1).int()

    np.save('data/MetroPT/리뉴얼/정상데이터/plot/Transformer_layer2/normal_loss2.npy', loss)
    np.save('data/MetroPT/리뉴얼/정상데이터/plot/Transformer_layer2/normal_pred2.npy', y_pred)
    np.save('data/MetroPT/리뉴얼/정상데이터/plot/Transformer_layer2/normal_input2.npy', input)
    print("loss.shape" ,loss.shape)
    print("y_pred.shape" ,y_pred.shape)
    print("input.shape" ,input.shape)