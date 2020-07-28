#!/usr/bin/env python
# coding: utf-8


from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


fix_seed(2020)


class LSTMCLF(nn.Module):
    def __init__(self, seq_embedding_features, statistics_features,
                 seq_statistics_features, seq_len, device):
        super(LSTMCLF, self).__init__()

        self.seq_embedding_features = seq_embedding_features
        self.statistics_features = statistics_features
        self.seq_statistics_features = seq_statistics_features

        self.seq_len = seq_len

        self.seq_statistics_size = len(seq_statistics_features)
        self.statistics_size = len(statistics_features)

        self.device = device

        input_size = 0
        self.embeds = nn.ModuleDict()
        
        for f in self.seq_embedding_features:
            embedding_layer = nn.Embedding(
                self.seq_embedding_features[f]['nunique'],
                self.seq_embedding_features[f]['embedding_dim'])

            pretrained_weight = np.array(
                self.seq_embedding_features[f]['pretrained_embedding'])
            embedding_layer.weight.data.copy_(
                torch.from_numpy(pretrained_weight))
            embedding_layer.weight.requires_grad = False
            self.embeds[f] = embedding_layer
        
        # LSTM 层
        for f in self.seq_embedding_features:
            input_size += seq_embedding_features[f]['embedding_dim']
        input_size += self.seq_statistics_size


        self.lstm = nn.LSTM(input_size,
                            128,
                            batch_first=True,
                            num_layers=2,
                            bidirectional=True)

        # Attention 层
        attention_input_size = 128 * 2
        self.attention_output_size = attention_input_size
        self.Q_weight = nn.Linear(attention_input_size,
                                  self.attention_output_size)
        self.K_weight = nn.Linear(attention_input_size,
                                  self.attention_output_size)
        self.V_weight = nn.Linear(attention_input_size,
                                  self.attention_output_size)

        # DNN 层
        dnn_input_size = self.attention_output_size + attention_input_size + self.statistics_size

        self.linears = nn.Sequential(nn.Linear(dnn_input_size, 1024),
                                     nn.LeakyReLU(), nn.BatchNorm1d(1024),
                                     nn.Linear(1024, 256), nn.LeakyReLU(),
                                     nn.BatchNorm1d(256), nn.Linear(256, 64),
                                     nn.LeakyReLU(), nn.BatchNorm1d(64),
                                     nn.Linear(64, 16), nn.LeakyReLU(),
                                     nn.BatchNorm1d(16), nn.Dropout(0.1))

        # age 输出层
        self.age_output = nn.Linear(16, 10)

    def forward(self, seq_id_list, statistics_input, statistics_seq_input_list,
                seq_lengths):
        batch_size = seq_id_list[0].shape[0]

        # 序列 id Embedding
        seq_feature_list = []
        for i, seq_id in enumerate(seq_id_list):
            feature_name = list(self.seq_embedding_features.keys())[i]
            embeddings = self.embeds[feature_name](seq_id.to(self.device))
            seq_feature_list.append(embeddings)

        seq_input = torch.cat(seq_feature_list, 2)
        seq_input = F.dropout2d(seq_input, 0.1, training=self.training)

        # LSTM
        seq_output, _ = self.lstm(seq_input)
        # mask padding
        mask = torch.zeros(seq_output.shape).to(self.device)
        for idx, seqlen in enumerate(seq_lengths):
            mask[idx, :seqlen] = 1

        seq_output = seq_output * mask
        lstm_output_max, _ = torch.max(seq_output, dim=1)

        # Attention
        Q = self.Q_weight(seq_output)
        K = self.K_weight(seq_output)
        V = self.V_weight(seq_output)

        tmp = torch.bmm(Q, K.transpose(1, 2))
        tmp = tmp / np.sqrt(self.attention_output_size)
        w = torch.softmax(tmp, 2)
        att_output = torch.bmm(w, V)
        att_output = att_output * mask
        att_max_output, _ = torch.max(att_output, dim=1)

        # 拼接统计特征
        cat_output = torch.cat(
            [att_max_output, lstm_output_max, statistics_input], 1)

        # DNN
        dnn_output = self.linears(cat_output)
        age_output = self.age_output(dnn_output)

        return age_output

    def set(self, criterion_age, optimizer, scheduler, early_stopping):
        self.criterion_age = criterion_age
        self.optimizer = optimizer
        self.scheduler = scheduler
        if early_stopping is None:
            self.early_stopping = EarlyStopping(
                file_name='model/age_checkpoint.pt', patience=10, verbose=True)
        else:
            self.early_stopping = early_stopping

#         self.set_embedding()

    def set_embedding(self):
        for f in self.seq_embedding_features:
            embedding_layer = nn.Embedding(
                self.seq_embedding_features[f]['nunique'],
                self.seq_embedding_features[f]['embedding_dim'])

            pretrained_weight = np.array(
                self.seq_embedding_features[f]['pretrained_embedding'])
            embedding_layer.weight.data.copy_(
                torch.from_numpy(pretrained_weight))
            embedding_layer.weight.requires_grad = False
            self.embeds[f] = embedding_layer

    def gen_data(self, data):
        # 序列 id 特征
        seq_id_list = []
        for f in self.seq_embedding_features.keys():
            vectorized_seqs = data[f + '_seq'].values

            seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))
            #             seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max()))

            #             for idx, (seq,
            #                       seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
            #                 seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

            #             seq_tensor = seq_tensor.long()
            #             seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
            seq = [torch.from_numpy(v) for v in vectorized_seqs]
            seq_tensor = pad_sequence(seq, batch_first=True, padding_value=0)
            seq_tensor = seq_tensor.long()
            seq_id_list.append(seq_tensor)

        # 统计特征
        statistics_input = data[self.statistics_features].values
        statistics_input = torch.Tensor(statistics_input).to(self.device)

        # 序列统计特征
        seq_statistics_list = []
        for f in self.seq_statistics_features:
            seq_statistics_input = data[f].values
            seq_statistics_input = torch.Tensor(seq_statistics_input).to(
                self.self.device)
            seq_statistics_list.append(seq_statistics_input)

        y_age = data['age'].values
        y_age = torch.LongTensor(y_age).to(self.device)

        return seq_id_list, statistics_input, seq_statistics_list, y_age, seq_lengths

    def model_train(self, train_data, val_data, epoches, batch_size):
        data_size = train_data.shape[0]
        n_batches = math.ceil(data_size / batch_size)

        best_age_acc = 0

        for epoch in range(epoches):
            model.train()

            train_loss_list = []
            for batch in tqdm(range(n_batches),
                              desc='epoch:{}/{}'.format(epoch, epoches)):
                start = batch * batch_size
                end = min((batch + 1) * batch_size, data_size)
                bs = end - start

                seq_id_list, statistics_input, seq_statistics_list, y_age, seq_lengths = self.gen_data(
                    train_data.iloc[start:end])

                pred_age = model(seq_id_list, statistics_input,
                                 seq_statistics_list, seq_lengths)

                loss = self.criterion_age(pred_age, y_age)

                train_loss_list.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                del seq_id_list, statistics_input, seq_statistics_list, y_age, loss
                torch.cuda.empty_cache()

            train_loss = np.mean(train_loss_list)
            _, val_loss, val_age_acc = self.model_predict(val_data,
                                                          batch_size,
                                                          log=True)
            if self.scheduler:
                self.scheduler.step(val_age_acc)

            if val_age_acc > best_age_acc:
                best_age_acc = val_age_acc

            print(
                'epoch {}/{} train_loss: {:.5f}, val_loss: {:.5f}, val_age_acc: {:.5f}'
                .format(epoch + 1, epoches, train_loss, val_loss, val_age_acc))

            self.early_stopping(val_age_acc, model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        return best_age_acc

    def model_predict(self, data, batch_size, log):
        model.eval()

        data_size = data.shape[0]
        n_batches = math.ceil(data_size / batch_size)

        if log:
            age_acc_list = []
            loss_list = []

        oof_pred_age = np.zeros((data.shape[0], 10))

        for batch in range(n_batches):
            start = batch * batch_size
            end = min((batch + 1) * batch_size, data_size)
            bs = end - start

            seq_id_list, statistics_input, seq_statistics_list, y_age, seq_lengths = self.gen_data(
                data.iloc[start:end])

            pred_age = model(seq_id_list, statistics_input,
                             seq_statistics_list, seq_lengths)

            oof_pred_age[start:end] = pred_age.cpu().data.numpy()

            del seq_id_list, statistics_input, seq_statistics_list

            if log:
                loss = self.criterion_age(pred_age, y_age)

                pred_age_cat = torch.max(pred_age, 1)[1].cpu().data.numpy()

                age_accuracy = float(
                    (pred_age_cat == y_age.cpu().data.numpy()
                     ).astype(int).sum()) / float(y_age.shape[0])

                age_acc_list.append(age_accuracy)
                loss_list.append(loss.item())

                del y_age, loss, pred_age

            torch.cuda.empty_cache()

        if log:
            return oof_pred_age, np.mean(loss_list), np.mean(age_acc_list)
        else:
            return oof_pred_age, None, None
