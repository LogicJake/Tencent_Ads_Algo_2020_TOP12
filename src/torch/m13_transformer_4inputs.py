#!/usr/bin/env python
# coding: utf-8


from transformers.modeling_bert import BertEmbeddings, BertEncoder
from transformers.configuration_bert import BertConfig

import torch
import torch.nn as nn
from torch.nn import functional as F
import math



def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

fix_seed(2020)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class Model(nn.Module):
    def __init__(self, embeddings, device):
        super(Model, self).__init__()
        self.device = device
        
        cid_emb_size = embeddings[0].shape[1]
        creative_id_embedding = nn.Embedding(embeddings[0].shape[0], cid_emb_size)
        creative_id_embedding.weight.data.copy_(torch.from_numpy(embeddings[0]))
        creative_id_embedding.weight.requires_grad = False
        self.creative_id_embedding = creative_id_embedding

        aid_emb_size = embeddings[1].shape[1]
        ad_id_embedding = nn.Embedding(embeddings[1].shape[0], aid_emb_size)
        ad_id_embedding.weight.data.copy_(torch.from_numpy(embeddings[1]))
        ad_id_embedding.weight.requires_grad = False
        self.ad_id_embedding = ad_id_embedding
        
        adv_emb_size = embeddings[2].shape[1]
        advertiser_id_embedding = nn.Embedding(embeddings[2].shape[0], adv_emb_size)
        advertiser_id_embedding.weight.data.copy_(torch.from_numpy(embeddings[2]))
        advertiser_id_embedding.weight.requires_grad = False
        self.advertiser_id_embedding = advertiser_id_embedding
        
        pid_emb_size = embeddings[3].shape[1]
        product_id_embedding = nn.Embedding(embeddings[3].shape[0], pid_emb_size)
        product_id_embedding.weight.data.copy_(torch.from_numpy(embeddings[3]))
        product_id_embedding.weight.requires_grad = False
        self.product_id_embedding = product_id_embedding
            
        hidden_size = cid_emb_size + aid_emb_size + adv_emb_size + pid_emb_size
        
        # transformer
        config = BertConfig(num_hidden_layers=3,
                            num_attention_heads=8,
                            hidden_size=hidden_size,
                            layer_norm_eps=1e-12,
                            hidden_dropout_prob=0.2,
                            attention_probs_dropout_prob=0.2,
                            hidden_act='mish')
        self.config = config
        self.bert_encoder = BertEncoder(config)
        
        # DNN 层
        self.linears = nn.Sequential(nn.Linear(config.hidden_size, 1024), Mish(), nn.BatchNorm1d(1024),
                                     nn.Linear(1024, 256), Mish(), nn.BatchNorm1d(256), 
                                     nn.Linear(256, 64), Mish(), nn.BatchNorm1d(64),
                                     nn.Linear(64, 16), Mish(), nn.BatchNorm1d(16), 
                                     nn.Dropout(0.1))

        # 输出层
        self.age_output = nn.Linear(16, 10)
        self.gender_output = nn.Linear(16, 2)

    def forward(self, seqs, seq_lengths):        
        # embedding
        cid_emb = self.creative_id_embedding(seqs[0])
        aid_emb = self.ad_id_embedding(seqs[1])
        advid_emb = self.advertiser_id_embedding(seqs[2])
        pid_emb = self.product_id_embedding(seqs[3])
        conc_emb = torch.cat([cid_emb, aid_emb, advid_emb, pid_emb], 2)
        
        # transformer
        head_mask = [None] * self.config.num_hidden_layers
        bert_ouput = self.bert_encoder(hidden_states=conc_emb, head_mask=head_mask)
        bert_ouput = bert_ouput[0]
        # mask padding
        mask = torch.zeros(bert_ouput.shape).to(self.device)
        for idx, seqlen in enumerate(seq_lengths):
            mask[idx, :seqlen] = 1
        bert_ouput = bert_ouput * mask
        bert_max, _ = torch.max(bert_ouput, dim=1)
        
        # DNN
        dnn_output = self.linears(bert_max)
        age_output = self.age_output(dnn_output)
        gender_output = self.gender_output(dnn_output)

        return age_output, gender_output

    def set(self, criterion_age,criterion_gender, optimizer, scheduler, early_stopping):
        self.criterion_age = criterion_age
        self.criterion_gender = criterion_gender

        self.optimizer = optimizer
        self.scheduler = scheduler
        if early_stopping is None:
            self.early_stopping = EarlyStopping(
                file_name='model/checkpoint.pt', patience=10, verbose=True)
        else:
            self.early_stopping = early_stopping

    def model_train(self, train_input, val_input, train_output, val_output, epoches, batch_size):
        data_size = train_input[0].shape[0]
        n_batches = math.ceil(data_size / batch_size)

        # 序列真实长度
        tmp = train_input[0]
        tmp[tmp < 0 ] = 0
        tmp[tmp > 0] = 1
        seq_lengths = tmp.sum(axis=1)
        
        best_age_acc = 0
        best_gender_acc = 0
        best_acc = 0
        for epoch in range(epoches):
            model.train()

            train_loss_list = []
            for batch in tqdm(range(n_batches),
                              desc='epoch:{}/{}'.format(epoch, epoches)):
                start = batch * batch_size
                end = min((batch + 1) * batch_size, data_size)
                bs = end - start

                batch_creative_id_seqs = train_input[0][start:end]
                batch_ad_id_seqs = train_input[1][start:end]
                batch_advertiser_id_seqs = train_input[2][start:end]
                batch_product_id_seqs = train_input[3][start:end]
                batch_creative_id_seqs = torch.LongTensor(batch_creative_id_seqs).to(self.device)
                batch_ad_id_seqs = torch.LongTensor(batch_ad_id_seqs).to(self.device)
                batch_advertiser_id_seqs = torch.LongTensor(batch_advertiser_id_seqs).to(self.device)
                batch_product_id_seqs = torch.LongTensor(batch_product_id_seqs).to(self.device)

                y_age = train_output[0][start:end]
                y_gender = train_output[1][start:end]
                y_age = torch.LongTensor(y_age).to(self.device)
                y_gender = torch.LongTensor(y_gender).to(self.device)
                
                batch_seq_lengths = seq_lengths[start:end]

                pred_age, pred_gender = model([batch_creative_id_seqs, batch_ad_id_seqs, batch_advertiser_id_seqs, batch_product_id_seqs], batch_seq_lengths)
                loss_age = self.criterion_age(pred_age, y_age)
                loss_gender = self.criterion_gender(pred_gender, y_gender)
                loss = loss_age + 0.1 * loss_gender
                train_loss_list.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                del batch_creative_id_seqs, batch_ad_id_seqs, batch_advertiser_id_seqs, batch_product_id_seqs, y_age, y_gender, loss, loss_age, loss_gender, pred_age, pred_gender
                torch.cuda.empty_cache()

            train_loss = np.mean(train_loss_list)

            # 预测验证集，计算指标
            model.eval()
            with torch.no_grad():
                val_pred_age, val_pred_gender = self.model_predict(val_input, batch_size)
 
                y_age = val_output[0]
                y_gender = val_output[1]
                y_age = torch.LongTensor(y_age).to(self.device)
                y_gender = torch.LongTensor(y_gender).to(self.device)

                loss_age = self.criterion_age(torch.from_numpy(val_pred_age).to(self.device), y_age)
                loss_gender = self.criterion_gender(torch.from_numpy(val_pred_gender).to(self.device), y_gender)
                val_loss = loss_age + 0.1 * loss_gender
                val_loss = val_loss.item()

                val_age_acc = accuracy_score(val_output[0], np.argmax(val_pred_age, axis=1))
                val_gender_acc = accuracy_score(val_output[1], np.argmax(val_pred_gender, axis=1))
            
            if self.scheduler:
                self.scheduler.step(val_age_acc)
            
            val_acc = val_age_acc + val_gender_acc
            if val_acc > best_acc:
                best_acc = val_acc
                best_age_acc = val_age_acc
                best_gender_acc = val_gender_acc

            print(
                'epoch {}/{} train_loss: {:.5f}, val_loss: {:.5f}, val_age_acc: {:.5f}, val_gender_acc: {:.5f}'
                .format(epoch + 1, epoches, train_loss, val_loss, val_age_acc, val_gender_acc))

            self.early_stopping(val_acc, model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        return best_acc, best_age_acc, best_gender_acc

    def model_predict(self, input_data, batch_size):
        model.eval()
        
        # 序列真实长度
        tmp = input_data[0]
        tmp[tmp < 0 ] = 0
        tmp[tmp > 0] = 1
        seq_lengths = tmp.sum(axis=1)
        
        data_size = input_data[0].shape[0]
        n_batches = math.ceil(data_size / batch_size)

        oof_pred_age = np.zeros((data_size, 10))
        oof_pred_gender = np.zeros((data_size, 2))

        for batch in range(n_batches):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, data_size)
            bs = end - start

            batch_creative_id_seqs = input_data[0][start:end]
            batch_ad_id_seqs = input_data[1][start:end]
            batch_advertiser_id_seqs = input_data[2][start:end]
            batch_product_id_seqs = input_data[3][start:end]
            batch_creative_id_seqs = torch.LongTensor(batch_creative_id_seqs).to(self.device)
            batch_ad_id_seqs = torch.LongTensor(batch_ad_id_seqs).to(self.device)
            batch_advertiser_id_seqs = torch.LongTensor(batch_advertiser_id_seqs).to(self.device)
            batch_product_id_seqs = torch.LongTensor(batch_product_id_seqs).to(self.device)

            batch_seq_lengths = seq_lengths[start:end]
            
            pred_age, pred_gender = model([batch_creative_id_seqs, batch_ad_id_seqs, batch_advertiser_id_seqs, batch_product_id_seqs],
                                         batch_seq_lengths)

            oof_pred_age[start:end] = pred_age.cpu().data.numpy()
            oof_pred_gender[start:end] = pred_gender.cpu().data.numpy()

            del batch_creative_id_seqs, batch_ad_id_seqs, batch_advertiser_id_seqs, batch_product_id_seqs
            torch.cuda.empty_cache()
        return oof_pred_age, oof_pred_gender


