import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class ProtoHATT(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, shots, hidden_size=230):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.drop = nn.Dropout()
        
        # for instance-level attention
        self.fc = nn.Linear(hidden_size, hidden_size, bias=True)
        # for feature-level attention
        self.conv1 = nn.Conv2d(1, 32, (shots, 1), padding=(shots // 2, 0))
        self.b1 = nn.Parameter(torch.randn(1, 32, 1, 1))
        self.conv1_gate = nn.Conv2d(1, 32, (shots, 1), padding=(shots // 2, 0))
        self.c1 = nn.Parameter(torch.randn(1, 32, 1, 1))

        self.conv2 = nn.Conv2d(32, 64, (shots, 1), padding=(shots // 2, 0))
        self.b2 = nn.Parameter(torch.randn(1, 64, 1, 1))
        self.conv2_gate = nn.Conv2d(32, 64, (shots, 1), padding=(shots // 2, 0))
        self.c2 = nn.Parameter(torch.randn(1, 64, 1, 1))

        #self.conv3 = nn.Conv2d(64, 80, (shots, 1), padding=(shots // 2, 0))
        #self.conv4 = nn.Conv2d(80, 128, (shots, 1), padding=(shots // 2, 0))
        self.conv_final = nn.Conv2d(64, 1, (shots, 1), stride=(shots, 1))
        self.b3 = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.conv_final_gate = nn.Conv2d(64, 1, (shots, 1), stride=(shots, 1))
        self.c3 = nn.Parameter(torch.randn(1, 1, 1, 1))

        #self.linear = torch.nn.Linear(73600,230)
        self.pooling = torch.nn.AdaptiveMaxPool3d((1,1,230))

    def __dist__(self, x, y, dim, score=None):
        if score is None:
            return (torch.pow(x - y, 2)).sum(dim)
        else:
            return (torch.pow(x - y, 2) * score).sum(dim)

    def __batch_dist__(self, S, Q, score=None):
        return self.__dist__(S, Q.unsqueeze(2), 3, score)

    def forward(self, support, query, N, K, Q, flag=0):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''
        support = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        query = self.sentence_encoder(query) # (B * N * Q, D)
        support = support.view(-1, N, K, self.hidden_size) # (B, N, K, D)
        query = query.view(-1, N * Q, self.hidden_size) # (B, N * Q, D)

        B = support.size(0) # Batch size
        NQ = query.size(1) # Num of instances for each batch in the query set
        
        seq_len = 128
        # feature-level attention
        fea_att_score = support.view(B * N, 1, K, self.hidden_size) # (B * N, 1, K, D)
        
        fea_att_score = F.relu(self.conv1(fea_att_score)) # (B * N, 32, K, D) 
        fea_att_score = F.relu(self.conv2(fea_att_score)) # (B * N, 64, K, D)
        #fea_att_score = F.relu(self.conv3(fea_att_score)) # (B * N, 80, K, D)
        #fea_att_score = F.relu(self.conv4(fea_att_score)) # (B * N, 128, K, D)

        fea_att_score = self.drop(fea_att_score)
        fea_att_score = self.conv_final(fea_att_score) # (B * N, 1, 1, D)
        fea_att_score = F.relu(fea_att_score)

        fea_att_score = self.drop(fea_att_score)
        fea_att_score = self.pooling(fea_att_score)

        '''
        fea_att_score_origin = self.conv1(fea_att_score) # (B * N, 32, K, D) 
        fea_att_score_origin += self.b1.repeat(1, 1, 5, 1)
        fea_att_score_gate = self.conv1_gate(fea_att_score) # (B * N, 32, K, D) 
        fea_att_score_gate += self.c1.repeat(1, 1, 5, 1)
        fea_att_score = fea_att_score_origin * F.sigmoid(fea_att_score_gate)

        fea_att_score_origin = self.conv2(fea_att_score) # (B * N, 64, K, D)
        fea_att_score_origin += self.b2.repeat(1, 1, 5, 1)
        fea_att_score_gate = self.conv2_gate(fea_att_score) # (B * N, 32, K, D) 
        fea_att_score_gate += self.c2.repeat(1, 1, 5, 1)
        fea_att_score = fea_att_score_origin * F.sigmoid(fea_att_score_gate)

        fea_att_score = self.drop(fea_att_score)
        fea_att_score_origin = self.conv_final(fea_att_score) # (B * N, 1, 1, D)
        fea_att_score_origin += self.b3.repeat(1, 1, 1, 1)
        fea_att_score_gate = self.conv_final_gate(fea_att_score) # (B * N, 1, 1, D)
        fea_att_score_gate += self.c3.repeat(1, 1, 1, 1)
        fea_att_score = fea_att_score_origin * F.sigmoid(fea_att_score_gate)
        #fea_att_score = F.relu(fea_att_score)
        #fea_att_score = self.drop(fea_att_score)
        fea_att_score = self.pooling(fea_att_score)
        '''

        #if flag == 0:
            #fea_att_score = F.relu(self.conv2(fea_att_score)) # (B * N, 64, K, D)
            #fea_att_score = F.relu(self.conv3(fea_att_score)) # (B * N, 80, K, D)
            #fea_att_score = F.relu(self.conv4(fea_att_score)) # (B * N, 128, K, D)
         #   fea_att_score = self.drop(fea_att_score)
          #  fea_att_score = self.conv_final(fea_att_score) # (B * N, 1, 1, D)
           # fea_att_score = F.relu(fea_att_score)
        #else:
            #fea_att_score = self.drop(fea_att_score)
            #fea_att_score = self.pooling(fea_att_score)
       
        fea_att_score = fea_att_score.view(B, N, self.hidden_size).unsqueeze(1) # (B, 1, N, D)

        # instance-level attention 
        support = support.unsqueeze(1).expand(-1, NQ, -1, -1, -1) # (B, NQ, N, K, D)
        support_for_att = self.fc(support) 
        query_for_att = self.fc(query.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, K, -1))
        ins_att_score = F.softmax(torch.tanh(support_for_att * query_for_att).sum(-1), dim=-1) # (B, NQ, N, K)
        support_proto = (support * ins_att_score.unsqueeze(4).expand(-1, -1, -1, -1, self.hidden_size)).sum(3) # (B, NQ, N, D)

        # Prototypical Networks 
        logits = -self.__batch_dist__(support_proto, query, fea_att_score)
        _, pred = torch.max(logits.view(-1, N), 1)
        return logits, pred
    
    
    
