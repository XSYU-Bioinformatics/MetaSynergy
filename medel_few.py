import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch.nn.utils.convert_parameters import (vector_to_parameters, parameters_to_vector)


class Feature_embedding_network(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, output_dimd=128, output_dimc=256, dropout=0.2):
        super(Feature_embedding_network, self).__init__()

        # Drugs
        self.conv1 = GCNConv(num_features_xd, num_features_xd * 2)
        self.conv2 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.conv3 = GCNConv(num_features_xd * 4, num_features_xd * 2)

        # Cell lines
        self.gconv1 = nn.Conv2d(1, 32, 7, 1, 1)
        self.norm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.15)
        self.gconv2 = nn.Conv2d(32, 64, 5, 1, 1)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.gconv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.gconv4 = nn.Conv2d(128, 64, 3, 1, 1)

        # Aggregation
        self.fc_g1 = torch.nn.Linear(num_features_xd * 2, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dimd)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_features_xd * 2)

        self.fcc1 = nn.Linear(64 * 25, output_dimc)
        self.normf = nn.BatchNorm1d(2 * output_dimd + output_dimc)

    def forward(self, data_train, data0_train, task_batch):
        # Drug A
        x1_train, edge_index1_train, batch1_train = data_train['x'], data_train['edge_index'], data_train['batch']
        x1 = self.conv1(x1_train, edge_index1_train)
        x11 = self.relu(x1)
        x11 = self.conv2(x11, edge_index1_train)
        x11 = self.relu(x11)
        x11 = self.conv3(x11, edge_index1_train)
        x1 = x1 + x11
        x1 = self.norm(x1)
        x1 = self.relu(x1)
        x1 = gmp(x1, batch1_train)

        # Drug B
        x2_train, edge_index2_train, batch2_train = data0_train['x'], data0_train['edge_index'], data0_train['batch']
        x2 = self.conv1(x2_train, edge_index2_train)
        x22 = self.relu(x2)
        x22 = self.conv2(x22, edge_index2_train)
        x22 = self.relu(x22)
        x22 = self.conv3(x22, edge_index2_train)
        x2 = x2 + x22
        x2 = self.norm(x2)
        x2 = self.relu(x2)
        x2 = gmp(x2, batch2_train)

        # Cell lines
        cell = data_train.c
        spc, h = cell.size()
        cell = cell.view(spc, int(h ** 0.5), int(h ** 0.5))
        cell = cell.unsqueeze(1)
        xt = self.pool1(F.relu(self.norm1(self.gconv1(cell))))
        xt = self.pool2(F.relu(self.norm2(self.gconv2(xt))))
        xt = F.relu(self.gconv3(xt))
        xt = F.relu(self.gconv4(xt))
        bs, kn, ce, ce1 = xt.size()  # batch size *(K+Q),channel, width, height
        xt = xt.view(-1, kn * ce * ce1)

        # Aggregation
        x1 = self.relu(self.fc_g1(x1))
        x1 = self.dropout(x1)
        x1 = self.fc_g2(x1)

        x2 = self.relu(self.fc_g1(x2))
        x2 = self.dropout(x2)
        x2 = self.fc_g2(x2)

        xt = self.relu(self.fcc1(xt))

        xc = torch.cat((x1, x2, xt), 1)
        xc = self.normf(xc)
        bskn, ce = xc.size()  # batch size*(k+q), embedding size(e.g.,512)
        embeddings = xc.view(task_batch, -1, ce)
        return embeddings


class Inference_model(nn.Module):
    def __init__(self):
        super(Inference_model, self).__init__()
        # self.fc1 = nn.Linear(512, 1024)
        # self.fc2 = nn.Linear(1024, 1)
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, task_embedding):
        # output1 = self.fc1(task_embedding)
        # output1 = self.relu(output1)
        # output1 = self.dropout(output1)
        # output = self.fc2(output1)  # [50,1]
        output1 = self.fc1(task_embedding)
        output1 = self.relu(output1)
        output1 = self.dropout(output1)
        output2 = self.fc2(output1)
        output2 = self.relu(output2)
        output2 = self.dropout(output2)
        output = self.fc3(output1)
        output = output.squeeze()  # 转换为 [50]
        return output


class HyperSynergy(nn.Module):
    def __init__(self, num_support, num_query):
        super(HyperSynergy, self).__init__()
        self.extractor = Feature_embedding_network()
        self.num_support = num_support  # K
        self.num_query = num_query  # Q
        self.prediction_network = Inference_model()
        self.optimizer = torch.optim.Adam(self.prediction_network.parameters(), lr=0.001)
        self.inner_l_rate = nn.Parameter(torch.FloatTensor([0.1]), requires_grad=True)
        self.finetuning_lr = nn.Parameter(torch.FloatTensor([0.001]), requires_grad=False)

    def update_params(self, loss, update_lr):
        grads = torch.autograd.grad(loss, self.prediction_network.parameters())
        return parameters_to_vector(grads), parameters_to_vector(self.prediction_network.parameters()) - parameters_to_vector(grads) * update_lr

    def run_batch(self, data_train, data0_train, batch_size, train=True):
        Nq = self.num_query
        NS = self.num_support
        NB = batch_size
        Y = data_train['y'].view(NB, -1)  # [50,90]
        support_labels = Y[:, :NS]  # [50, 50]
        target_label = Y[:, NS:] .reshape(-1)  # [2000]
        support_target_embeddings = self.extractor(data_train, data0_train, NB)  # [50,90,512]
        support_embedings = support_target_embeddings[:, :NS]   # [50,50,512]
        target_embedings = support_target_embeddings[:, NS:]  # [50,40,512]
        outputs = torch.Tensor().to(device="cuda:1")
        criterion = nn.MSELoss()
        # 保存当前协同预测网络的初始化参数
        old_params = parameters_to_vector(self.prediction_network.parameters())
        for task in range(NB):
            # 1. 从任务中获取数据
            for i in range(5):
                support_pred = self.prediction_network(support_embedings[task])
                support_loss = criterion(support_pred, support_labels[task])
                # 3. 计算梯度并更新协同预测网络的参数
                new_grad, new_params = self.update_params(support_loss, update_lr=self.finetuning_lr)
                vector_to_parameters(new_params, self.prediction_network.parameters())
                # 4. 使用任务学习率更新参数
            # 5. 在查询集上计算损失
            query_pred = self.prediction_network(target_embedings[task])
            # 6. 恢复模型参数到初始化状态
            vector_to_parameters(old_params, self.prediction_network.parameters())
            outputs = torch.cat([outputs, query_pred], dim=0)  # [2000]
        query_loss = criterion(outputs, target_label)
        self.prediction_network.zero_grad()
        query_loss.backward()
        # 7. 使用 Adam 优化器更新模型参数
        self.optimizer.step()

        return outputs, target_label

    def test_batch(self, data_train, data0_train, batch_size, train=True):
        Nq = self.num_query
        NS = self.num_support
        NB = batch_size
        Y = data_train['y'].view(NB, -1)  # [50,90]
        support_labels = Y[:, :NS]  # [50, 50]
        target_label = Y[:, NS:]  # [50, 50]
        target_label = target_label.reshape(-1)
        support_target_embeddings = self.extractor(data_train, data0_train, NB)  # [50,90,512]
        support_embedings = support_target_embeddings[:, :NS]   # [50,50,512]
        target_embedings = support_target_embeddings[:, NS:]  # [50,40,512]
        outputs = torch.Tensor().to(device="cuda:1")
        criterion = nn.MSELoss()
        # 保存当前协同预测网络的初始化参数
        old_params = parameters_to_vector(self.prediction_network.parameters())
        for task in range(NB):
            # 1. 从任务中获取数据
            for i in range(5):
                support_pred = self.prediction_network(support_embedings[task])
                support_loss = criterion(support_pred, support_labels[task])
                # 3. 计算梯度并更新协同预测网络的参数
                new_grad, new_params = self.update_params(support_loss, update_lr=self.finetuning_lr)
                vector_to_parameters(new_params, self.prediction_network.parameters())
            # 5. 在查询集上计算损失
            query_pred = self.prediction_network(target_embedings[task])
            vector_to_parameters(old_params, self.prediction_network.parameters())
            outputs = torch.cat([outputs, query_pred], dim=0)  # [2000]

        return outputs, target_label

