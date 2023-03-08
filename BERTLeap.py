
from IPython import get_ipython
# Old: dependencies and cloned repo
# New: Change directory to model 

# ####**0.2 Download pre-trained models and place them to data/pretrained/, you could choose from [VisualBERT](https://github.com/uclanlp/visualbert), [LXMERT](https://github.com/airsplay/lxmert), [UNITER](https://github.com/ChenRocks/UNITER).**

# ####**0.4 Download the visaul features extracted by BUTD. 36 2048-dimension visual feature is extracted from each CXR Image. We use this [implementation](https://github.com/airsplay/py-bottom-up-attention). This step will take a while (~1min). To save downloading time, you can also make a copy of this [shareable link](https://drive.google.com/file/d/1BFw0jc0j-ffT2PhI4CZeP3IJFZg3GxlZ/view?usp=sharing) to your own google drive and mount you colab to your gdrive.**

# ***0.4.1 Load visual features***



# In[ ]:


# feature_example, bbox_example, (img_w_example, img_h_example) = openI_v_f[openI.id.iloc[0]]


# In[ ]:


# feature_example.shape, bbox_example.shape, (img_w_example, img_h_example)


# ####**Now We have download all data, models, and dependencies. We are good to go!!!**
# **1. Change default arguments**
# 
# First, let's check it out!

# In[ ]:


from param import args


# In[ ]:

args.__dict__

args.batch_size = 18
args.epochs = 2
args.model = 'visualbert' # use visualbert
args.load_pretrained = '/content/Transformers-VQA/models/pretrained/visualbert.th' #load pretrained visualbert model
args.max_seq_length = 128 #truncate or pad report lengths to 128 subwords


# In[ ]:





# TODO: Create customized dataloader**

# In[ ]:


findings = list(openI.columns[1:-2])


# In[ ]:


from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
class OpenIDataset(Dataset):
  def __init__(self, df, vf, split, model = 'lxmert'):
    # train_test_split and prepare labels
    self.dataset = df[df['split'] == split]
    self.visual_features = vf
    self.id_list = self.dataset.id.tolist()
    self.report_list = self.dataset.TXT.tolist()
    self.findings_list = self.dataset.columns[1:-2]
    self.target_list = self.dataset[self.findings_list].to_numpy().astype(np.float32)
    self.model = model

  def __len__(self):
    return len(self.id_list)

  def __getitem__(self, item):
    cxr_id = self.id_list[item]
    target = self.target_list[item]
    boxes, feats, (img_w, img_h) = self.visual_features[cxr_id]
    report = self.report_list[item]
    if self.model == 'uniter':
      boxes = self._uniterBoxes(boxes)
    if self.model == 'lxmert':
      boxes[:, (0, 2)] /= img_w
      boxes[:, (1, 3)] /= img_h
    return cxr_id, feats, boxes, report, target

  def _uniterBoxes(self, boxes):#uniter requires a 7-dimensiom beside the regular 4-d bbox
    new_boxes = np.zeros((boxes.shape[0],7),dtype='float32')
    new_boxes = np.zeros((boxes.shape[0],7),dtype='float32')
    new_boxes[:,1] = boxes[:,0]
    new_boxes[:,0] = boxes[:,1]
    new_boxes[:,3] = boxes[:,2]
    new_boxes[:,2] = boxes[:,3]
    new_boxes[:,4] = new_boxes[:,3]-new_boxes[:,1] #w
    new_boxes[:,5] = new_boxes[:,2]-new_boxes[:,0] #h
    new_boxes[:,6]=new_boxes[:,4]*new_boxes[:,5] #area
    return new_boxes  


# In[ ]:


training = OpenIDataset(df = openI, vf = openI_v_f,  split='train', model = args.model)
testing = OpenIDataset(df = openI, vf = openI_v_f,  split='test', model = args.model)


# In[ ]:


train_loader = DataLoader(training, batch_size=args.batch_size,shuffle=True, num_workers=0,drop_last=True, pin_memory=True)
test_loader = DataLoader(testing, batch_size=128,shuffle=False, num_workers=0,drop_last=False, pin_memory=True)


# ####**3. Model, Optimizer, Loss Function, and Evaluation Function**

# In[ ]:


from vqa_model import VQAModel
#init model
model = VQAModel(num_answers = len(findings), model = args.model)


# In[ ]:


#load pretrained weights
model.encoder.load(args.load_pretrained)




import torch
loss = torch.nn.BCEWithLogitsLoss()


# In[ ]:


from src.optimization import BertAdam
optim = BertAdam(list(model.parameters()),lr=args.lr,warmup=0.1,t_total=len(train_loader)*args.epochs)
# t_total denotes total training steps
# batch_per_epoch = len(train_loader)
# t_total = int(batch_per_epoch * args.epochs)


# In[ ]:


# Evaluation function, we will report the AUC and accuray of each finding
def eval(target, pred):
    acc_list = []
    for i, d in enumerate(findings[:-1]): #normal is excluded
        acc = np.mean(target[:,i] == (pred[:,i]>=0.5))
        print(i,d,acc)
        acc_list.append(acc)
    print('Averaged: '+str(np.average(acc_list)))


# In[ ]:


sgmd = torch.nn.Sigmoid()


# ####**4. HIT and RUN**

# In[ ]:


from tqdm.notebook import tqdm

iter_wrapper = (lambda x: tqdm(x, total=len(train_loader))) if args.tqdm else (lambda x: x)
best_valid = 0
for epoch in range(args.epochs):
  epoch_loss = 0
  for i, (cxr_id, feats, boxes, report, target) in iter_wrapper(enumerate(train_loader)):
    model.train()
    optim.zero_grad()
    feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
    logit = model(feats, boxes, report)
    running_loss = loss(logit, target)
    running_loss = running_loss * logit.size(1)
    epoch_loss += running_loss
    running_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
    optim.step()
  print("Epoch "+str(epoch)+": Training Loss: "+str(epoch_loss/len(train_loader)))
  print('Evaluation: ')
  model.eval()
  logit_list, target_list = [], []
  iter_wrapper = (lambda x: tqdm(x, total=len(test_loader)))
  for i, (cxr_id, feats, boxes, report, target) in iter_wrapper(enumerate(test_loader)):
    target_list.append(target)
    with torch.no_grad():
      feats, boxes = feats.cuda(), boxes.cuda()
      logit = model(feats, boxes, report)
      logit_list.append(sgmd(logit).cpu().numpy())

  eval(np.concatenate(target_list,axis = 0), np.concatenate(logit_list,axis = 0))


# In[ ]:




