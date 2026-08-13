import pickle
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layer import *

from .model import BaseModel

from copy import deepcopy



class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, r=None, train=None, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        if r is not None:
            gates = F.softmax(logits / torch.sigmoid(r), dim=-1)
        else:
            gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x, r=None):
        gates = self.noisy_top_k_gating(x, r, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2) + x, expert_outputs, gates
    
class CrossMoEAdaptorLayer(nn.Module):
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(CrossMoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[2], layers[1]), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[2], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, y, r=None, train=None, noise_epsilon=1e-2):
        clean_logits = torch.einsum('bn,bkn->bk', x @ self.w_gate, y)  # (B, n_E)
        # clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        if r is not None:
            gates = F.softmax(logits / torch.sigmoid(r), dim=-1)
        else:
            gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x, y, r=None):
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D), (B, 1, D), ...]
        expert_outputs = torch.cat(expert_outputs, dim=-2) # [(B, n_E, D)]
        gates = self.noisy_top_k_gating(y, expert_outputs, r, self.training) # (B, n_E)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2) + x, expert_outputs, gates
    

import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layer import *

from .model import BaseModel

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, multi, dropout=0.1, modal_dims=None):
        super().__init__()
        self.modal_dims = modal_dims
        num_modals = 6
        if modal_dims is not None:
            self.in_projection_s = nn.ModuleList([nn.Linear(modal_dim, hidden_dim) for modal_dim in modal_dims])
        self.num_heads = multi
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // multi
        self.scale = self.head_dim ** -0.5
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        init_bias = torch.zeros(multi, num_modals, num_modals)
        self.bias = nn.Parameter(init_bias)
        self.dropout = nn.Dropout(dropout)
        if modal_dims is not None:
            self.out_projection = nn.ModuleList([nn.Linear(hidden_dim, modal_dim) for modal_dim in modal_dims])
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.idx = torch.tensor([
            [0, 2, 2, 1, 1, 3],
            [2, 0, 2, 1, 3, 1],
            [2, 2, 0, 3, 1, 1],
            [1, 1, 3, 0, 2, 2],
            [1, 3, 1, 2, 0, 2],
            [3, 1, 1, 2, 2, 0]
        ], dtype=torch.long)
    
    def forward(self, modals):
        batch_size = modals[0].size(0)
        
        if self.modal_dims is not None:
            aligned = [self.in_projection_s[i](modals[i]) for i in range(len(modals))]
        else:
            aligned = []
            V_list = []
            data_mean_list = []
            for data in modals:
                data_meaned = data - data.mean(dim=0, keepdim=True)
                data_mean_list.append(data_meaned)
                data_centered = data - data_meaned
                noise = 1e-6 * torch.randn_like(data_centered)
                try:
                    U, S, V = torch.pca_lowrank(data_centered + noise, q=self.hidden_dim)
                except Exception:
                    noise = 1e-3 * torch.randn_like(data_centered)
                    U, S, V = torch.pca_lowrank(data_centered + noise, q=self.hidden_dim)
                reduced = torch.mm(data_centered, V[:, :self.hidden_dim])
                V_list.append(V)
                aligned.append(reduced)
        
        attention_input = torch.stack(aligned, dim=1)
        Q = self.W_q(attention_input).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(attention_input).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(attention_input).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # scores += self.bias[:, self.idx].unsqueeze(0)  # (1, head_num, num_modalities, num_modalities)
        scores += self.bias.unsqueeze(0)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim) # (B, num_modalities, hidden_dim)
        output = self.out_proj(context)
        
        result = []
        if self.modal_dims is not None:
            for i in range(output.shape[1]):
                modal = output[:, i, :]
                modal_reconstructed = self.out_projection[i](modal)
                result.append(modal_reconstructed)
        else:
            for i in range(output.shape[1]):
                modal = output[:, i, :]
                modal_reconstructed = torch.mm(modal, V_list[i][:, :self.hidden_dim].t()) + data_mean_list[i]
                result.append(modal_reconstructed)
        return result, self.bias.data


class ModalFusionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, multi, img_dim, txt_dim, extra=False, img_dim_ori=None, txt_dim_ori=None, pool_dim=128, temperature=0.5):
        super(ModalFusionLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.multi = multi
        self.img_dim = img_dim
        self.text_dim = txt_dim
        self.extra = extra
        self.pool_dim = pool_dim
        self.temperature = temperature
        self.step = 0

        modal1 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(in_dim, out_dim)
            modal1.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal1_layers = nn.ModuleList(modal1)

        modal2 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.img_dim, out_dim)
            modal2.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal2_layers = nn.ModuleList(modal2)

        modal3 = []
        for _ in range(self.multi):
            do = nn.Dropout(p=0.2)
            lin = nn.Linear(self.text_dim, out_dim)
            modal3.append(nn.Sequential(do, lin, nn.ReLU()))
        self.modal3_layers = nn.ModuleList(modal3)

        if extra:
            modal4 = []
            for _ in range(self.multi):
                do = nn.Dropout(p=0.2)
                lin = nn.Linear(img_dim_ori, out_dim)
                modal4.append(nn.Sequential(do, lin, nn.ReLU()))
            self.modal4_layers = nn.ModuleList(modal4)

            modal5 = []
            for _ in range(self.multi):
                do = nn.Dropout(p=0.2)
                lin = nn.Linear(txt_dim_ori, out_dim)
                modal5.append(nn.Sequential(do, lin, nn.ReLU()))
            self.modal5_layers = nn.ModuleList(modal5)

            modal6 = []
            for _ in range(self.multi):
                do = nn.Dropout(p=0.2)
                lin = nn.Linear(256, out_dim)
                modal6.append(nn.Sequential(do, lin, nn.ReLU()))
            self.modal6_layers = nn.ModuleList(modal6)

        self.pool = nn.AdaptiveAvgPool1d(pool_dim)
        self.pool_multi = 3 if not self.extra else 6
        pool_gate = []
        for _ in range(self.pool_multi):
            pool_gate.append(nn.Parameter(torch.zeros((pool_dim, 1))))
        self.pool_gate = nn.ParameterList(pool_gate)
        # self.pool_gate = nn.Parameter(torch.zeros((pool_dim, 1)))
        self.ent_attn = nn.Linear(self.out_dim, 1, bias=False)
        # self.ent_attn = nn.Sequential(
        #     nn.Linear(self.out_dim, self.out_dim // 2),
        #     nn.Tanh(),
        #     nn.Linear(self.out_dim // 2, 1, bias=False))
        self.ent_attn.requires_grad_(True)

    def forward(self, modal1_emb, modal2_emb, modal3_emb, modal4_emb=None, modal5_emb=None, modal6_emb=None):
        # print(modal6_emb)
        # if self.extra:
        #     assert modal4_emb is not None, "modal4_emb should not be None"
        #     assert modal5_emb is not None, "modal5_emb should not be None"
        #     assert modal6_emb is not None, "modal6_emb should not be None"
        batch_size = modal1_emb.size(0)
        x_mm = []
        for i in range(self.multi):
            x_modal1 = self.modal1_layers[i](modal1_emb * F.sigmoid(self.pool(modal1_emb) @ self.pool_gate[0]))
            x_modal2 = self.modal2_layers[i](modal2_emb * F.sigmoid(self.pool(modal2_emb) @ self.pool_gate[1]))
            x_modal3 = self.modal3_layers[i](modal3_emb * F.sigmoid(self.pool(modal3_emb) @ self.pool_gate[2]))
            modals = [x_modal1, x_modal2, x_modal3]
            if modal4_emb is not None:
                x_modal4 = self.modal4_layers[i](modal4_emb * F.sigmoid(self.pool(modal4_emb) @ self.pool_gate[3]))
                modals.append(x_modal4)
            if modal5_emb is not None:
                x_modal5 = self.modal5_layers[i](modal5_emb * F.sigmoid(self.pool(modal5_emb) @ self.pool_gate[4]))
                modals.append(x_modal5)
            if modal6_emb is not None:
                x_modal6 = self.modal6_layers[i](modal6_emb * F.sigmoid(self.pool(modal6_emb) @ self.pool_gate[5]))
                modals.append(x_modal6)
            x_stack = torch.stack(modals, dim=1)
            
            attention_scores = self.ent_attn(x_stack).squeeze(-1)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            # test_weight = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(attention_weights.device)
            # attention_weights = attention_weights * (1 - self.temperature) + test_weight * self.temperature
            # attention_weights = attention_weights * (1 - self.temperature) + (1.0 / self.pool_multi) * self.temperature
            context_vectors = torch.sum(attention_weights.unsqueeze(-1) * x_stack, dim=1)
            x_mm.append(context_vectors)
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.out_dim)
        # x_mm = torch.relu(x_mm)
        # if self.training:
        #     self.step += 1
        #     if self.step % 1000 == 0:
        #         self.temperature = max(0.2, self.temperature - 0.05)
        return x_mm, attention_weights
    
    def relation_gated_fuse(self, modal1_emb, modal2_emb, modal3_emb, rel):
        batch_size = modal1_emb.size(0)
        x_mm = []
        for i in range(self.multi):
            x_modal1 = self.modal1_layers[i](modal1_emb)
            x_modal2 = self.modal2_layers[i](modal2_emb)
            x_modal3 = self.modal3_layers[i](modal3_emb)
            x_stack = torch.stack((x_modal1, x_modal2, x_modal3), dim=1)
            attention_scores = self.ent_attn(x_stack).squeeze(-1)
            attention_weights = torch.softmax(attention_scores / rel, dim=-1)
            context_vectors = torch.sum(attention_weights.unsqueeze(-1) * x_stack, dim=1)
            x_mm.append(context_vectors)
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.mean(1).view(batch_size, self.out_dim)
        x_mm = torch.relu(x_mm)
        return x_mm
    
    def gated_fusion(self, emb, rel):
        # emb: batch_size x dim
        # rel: batch_size x dim
        w = torch.sigmoid(emb * rel)
        return w * emb + (1 - w) * rel



class RelMoE(BaseModel):
    def __init__(self, args):
        super(RelMoE, self).__init__(args)
        self.entity_embeddings = nn.Embedding(
            len(args.entity2id),
            args.dim,
            padding_idx=None
        )
        nn.init.xavier_normal_(self.entity_embeddings.weight)

        self.relation_embeddings = nn.Embedding(
            2 * len(args.relation2id), 
            args.r_dim, 
            padding_idx=None
        )
        nn.init.xavier_normal_(self.relation_embeddings.weight)

        if args.pre_trained:
            # Prefer GNN-produced embeddings if available
            ent_path = os.path.join('datasets', args.dataset, 'entity_embeddings_gnn.pth')
            rel_path = os.path.join('datasets', args.dataset, 'relation_embeddings_gnn.pth')
            if os.path.exists(ent_path) and os.path.exists(rel_path):
                ent_emb = torch.load(ent_path)
                rel_emb = torch.load(rel_path)
                # entities
                if ent_emb.size(0) != len(args.entity2id):
                    print(f"Warning: GNN entity embedding count {ent_emb.size(0)} != entity2id count {len(args.entity2id)}")
                if ent_emb.size(1) == args.dim:
                    self.entity_embeddings = nn.Embedding.from_pretrained(ent_emb.float(), freeze=False)
                else:
                    # project to model dim then use as init
                    self.entity_proj = nn.Linear(ent_emb.size(1), args.dim)
                    with torch.no_grad():
                        init_ent = self.entity_proj(ent_emb.float())
                    self.entity_embeddings = nn.Embedding.from_pretrained(init_ent, freeze=False)
                # relations: GNN saved per-relation embeddings (no inverse); model expects 2*relations (inverse concatenated)
                if rel_emb.size(1) == args.r_dim:
                    rel_cat = torch.cat((rel_emb.float(), -rel_emb.float()), dim=0)
                else:
                    self.rel_proj = nn.Linear(rel_emb.size(1), args.r_dim)
                    with torch.no_grad():
                        rproj = self.rel_proj(rel_emb.float())
                    rel_cat = torch.cat((rproj, -rproj), dim=0)
                self.relation_embeddings = nn.Embedding.from_pretrained(rel_cat, freeze=False)
            else:
                # fallback to original GAT pickle initialization
                self.entity_embeddings = nn.Embedding.from_pretrained(
                    torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_entity_vec.pkl', 'rb'))).float(), freeze=False)
                self.relation_embeddings = nn.Embedding.from_pretrained(torch.cat((
                    torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float(),
                    -1 * torch.from_numpy(pickle.load(open('datasets/' + args.dataset + '/gat_relation_vec.pkl', 'rb'))).float()), dim=0), freeze=False)

        self.rel_gate = nn.Embedding(2 * len(args.relation2id), 1, padding_idx=None)

        # print(args)
        if args.dataset == "DB15K":
            img_pool = torch.nn.AvgPool2d(4, stride=4)
            img = img_pool(args.img.to(self.device).view(-1, 64, 64))
            img = img.view(img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)
        elif "MKG" in args.dataset:
            # multi-modal information for MKG
            d_img = torch.load('datasets/' + args.dataset + '/diffusion_image_embeddings.pt')
            img_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            d_img = img_pool(d_img.to(self.device).view(-1, 32, 64))
            d_img = d_img.view(d_img.size(0), -1)

            # Prefer GNN-produced modality embeddings if available
            # gnn_img_path = os.path.join('datasets', args.dataset, 'img_embeddings_gnn.pth')
            # gnn_txt_path = os.path.join('datasets', args.dataset, 'text_embeddings_gnn.pth')
            # gnn_rel_path = os.path.join('datasets', args.dataset, 'relation_embeddings_gnn.pth')
            gnn_img_path = ""
            gnn_txt_path = ""
            gnn_rel_path = ""

            if os.path.exists(gnn_img_path):
                img = torch.load(gnn_img_path).to(self.device)
            else:
                img = args.img.to(self.device).view(args.img.size(0), -1)

            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            if os.path.exists(gnn_txt_path):
                txt = torch.load(gnn_txt_path).to(self.device)
            else:
                txt = txt_pool(args.desp.to(self.device).view(-1, 12, 32))
                txt = txt.view(txt.size(0), -1)

            # If a GNN relation embedding exists, use it to initialize modality relation embeddings as well
            rel_cat = None
            if os.path.exists(gnn_rel_path):
                rel_emb_gnn = torch.load(gnn_rel_path)
                # rel_emb_gnn may already be concatenated (2 * num_rel_types) or single (num_rel_types)
                if rel_emb_gnn.size(0) == 2 * len(args.relation2id):
                    rel_cat = rel_emb_gnn.float()
                elif rel_emb_gnn.size(0) == len(args.relation2id):
                    rel_cat = torch.cat((rel_emb_gnn.float(), -rel_emb_gnn.float()), dim=0)
                else:
                    print(f"Warning: Unexpected GNN relation count {rel_emb_gnn.size(0)}; expected {len(args.relation2id)} or {2 * len(args.relation2id)}")
                    # try to project or fallback
                    if rel_emb_gnn.size(0) > len(args.relation2id):
                        rel_cat = rel_emb_gnn.float()
            else:
                rel_cat = None
        elif "TIVA" in args.dataset:
            img_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            img = img_pool(args.img.to(self.device).view(-1, 32, 64))
            img = img.view(img.size(0), -1)
            txt = args.desp.to(self.device)
            txt = txt.view(txt.size(0), -1)
        elif "Kuai" in args.dataset:
            img_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            img = img_pool(args.img.to(self.device).view(-1, 12, 64))
            img = img.view(img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)
        elif "WN9" in args.dataset:
            img_pool = torch.nn.AvgPool2d(4, stride=4)
            img = img_pool(args.img.to(self.device).view(-1, 64, 64))
            img = img.view(img.size(0), -1)
            img = torch.tensor(img).to(torch.float32)
            txt = args.desp.to(self.device)
            txt = txt.view(txt.size(0), -1)
            txt = torch.tensor(txt).to(torch.float32)
        elif "FB15K-237" in args.dataset:
            img_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            img = img_pool(args.img.to(self.device).view(-1, 12, 64))
            img = img.view(img.size(0), -1)
            txt_pool = torch.nn.AdaptiveAvgPool2d(output_size=(4, 64))
            txt = txt_pool(args.desp.to(self.device).view(-1, 12, 64))
            txt = txt.view(txt.size(0), -1)

        # initialize modality entity embeddings
        self.img_entity_embeddings = nn.Embedding.from_pretrained(img, freeze=False)
        self.d_img_entity_embeddings = nn.Embedding.from_pretrained(d_img, freeze=False)
        # text embeddings (may come from GNN or pooled descriptors)
        self.txt_entity_embeddings = nn.Embedding.from_pretrained(txt, freeze=False)

        # initialize modality relation embeddings; prefer GNN-trained relations when available
        if rel_cat is not None:
            # project if necessary
            if rel_cat.size(1) != args.r_dim:
                rel_proj = nn.Linear(rel_cat.size(1), args.r_dim)
                with torch.no_grad():
                    rel_cat_proj = rel_proj(rel_cat)
                rel_init = rel_cat_proj
            else:
                rel_init = rel_cat
            self.img_relation_embeddings = nn.Embedding.from_pretrained(rel_init, freeze=False)
            self.img_relation_embeddings_cross = nn.Embedding.from_pretrained(rel_init, freeze=False)
            self.txt_relation_embeddings = nn.Embedding.from_pretrained(rel_init, freeze=False)
            self.txt_relation_embeddings_cross = nn.Embedding.from_pretrained(rel_init, freeze=False)
            self.d_img_relation_embeddings = nn.Embedding.from_pretrained(rel_init, freeze=False)
        else:
            # fallback to random init if no GNN relations
            self.img_relation_embeddings = nn.Embedding(
                2 * len(args.relation2id),
                args.r_dim, 
                padding_idx=None
            )
            nn.init.xavier_normal_(self.img_relation_embeddings.weight)
            self.img_relation_embeddings_cross = nn.Embedding(
                2 * len(args.relation2id),
                args.r_dim, 
                padding_idx=None
            )
            nn.init.xavier_normal_(self.img_relation_embeddings_cross.weight)
            self.txt_relation_embeddings = nn.Embedding(
                2 * len(args.relation2id),
                args.r_dim,
                padding_idx=None
            )
            nn.init.xavier_normal_(self.txt_relation_embeddings.weight)
            self.txt_relation_embeddings_cross = nn.Embedding(
                2 * len(args.relation2id),
                args.r_dim,
                padding_idx=None
            )
            nn.init.xavier_normal_(self.txt_relation_embeddings_cross.weight)
            self.d_img_relation_embeddings = nn.Embedding(
                2 * len(args.relation2id),
                args.r_dim,
                padding_idx=None
            )
            nn.init.xavier_normal_(self.d_img_relation_embeddings.weight)

        # Score Functions
        self.dim = args.dim
        self.img_dim = self.img_entity_embeddings.weight.data.shape[1]
        self.txt_dim = self.txt_entity_embeddings.weight.data.shape[1]
        self.clip_dim = args.clip_dim
        self.d_image_dim = args.diffusion_image_dim
        self.fuse_out_dim = self.dim
        # Score function layers
        self.TuckER_S = TuckERLayer(args.dim, args.r_dim)
        # self.TuckER_I = TuckERLayer(self.clip_dim, args.r_dim)
        # self.TuckER_D = TuckERLayer(self.clip_dim, args.r_dim)
        self.TuckER_I = TuckERLayer(self.img_dim, args.r_dim)
        self.TuckER_D = TuckERLayer(self.txt_dim, args.r_dim)
        self.TuckER_IS = TuckERLayer(self.img_dim, args.r_dim)
        self.TuckER_DS = TuckERLayer(self.txt_dim, args.r_dim)
        self.TuckER_DiffusionI = TuckERLayer(self.d_image_dim, args.r_dim)
        self.TuckER_MM = TuckERLayer(args.dim, self.fuse_out_dim)
        # Multi-modal fusion layers

        self.visual_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.img_dim, self.img_dim])
        self.text_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.txt_dim, self.txt_dim])
        self.structure_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.dim, self.dim])
        self.mm_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.fuse_out_dim, self.fuse_out_dim])

        self.visual_structure_moe = CrossMoEAdaptorLayer(n_exps=args.n_exp, layers=[self.img_dim, self.img_dim, self.dim])
        self.text_structure_moe = CrossMoEAdaptorLayer(n_exps=args.n_exp, layers=[self.txt_dim, self.txt_dim, self.dim])

        self.diffusion_visual_moe = MoEAdaptorLayer(n_exps=args.n_exp, layers=[self.d_image_dim, self.d_image_dim])
    
        # self.img_W = nn.Parameter(torch.randn(self.img_dim, self.clip_dim))
        # self.txt_W = nn.Parameter(torch.randn(self.txt_dim, self.clip_dim))

        self.fuse_e = ModalFusionLayer(
            in_dim=args.dim,
            out_dim=self.fuse_out_dim,
            multi=2,
            img_dim=self.img_dim,
            txt_dim=self.txt_dim,
            extra=True,
            img_dim_ori=self.img_dim,
            txt_dim_ori=self.txt_dim
        )
        self.fuse_r = ModalFusionLayer(
            in_dim=args.r_dim,
            out_dim=self.fuse_out_dim,
            multi=2,
            img_dim=args.r_dim,
            txt_dim=args.r_dim,
            extra=True,
            img_dim_ori=args.r_dim,
            txt_dim_ori=args.r_dim
        )
        self.bias = nn.Parameter(torch.zeros(len(args.entity2id)))
        self.bceloss = nn.BCELoss()
        # self.logit_scale = nn.Parameter(torch.tensor(0.0))
        
        # self.attention = MultiHeadAttentionLayer(hidden_dim=512, multi=4, modal_dims=[self.dim, self.img_dim, self.txt_dim, self.img_dim, self.txt_dim, self.d_image_dim])
        self.attention = MultiHeadAttentionLayer(hidden_dim=64, multi=4, modal_dims=None)

        
    def forward(self, batch_inputs):
        head = batch_inputs[:, 0]
        relation = batch_inputs[:, 1]
        rel_gate = self.rel_gate(relation)
        e_embed_head = self.entity_embeddings(head)
        e_img_embed_head = self.img_entity_embeddings(head)
        e_txt_embed_head = self.txt_entity_embeddings(head)
        e_d_img_embed_head = self.d_img_entity_embeddings(head)

        e_embed, disen_str, atten_s = self.structure_moe(e_embed_head, rel_gate)
        r_embed = self.relation_embeddings(relation)

        e_img_embed_ori, disen_img, atten_i = self.visual_moe(e_img_embed_head, rel_gate)
        # e_img_embed = e_img_embed_ori @ self.img_W
        r_img_embed = self.img_relation_embeddings(relation)
        r_img_embed_cross = self.img_relation_embeddings_cross(relation)

        e_txt_embed_ori, disen_txt, atten_t = self.text_moe(e_txt_embed_head, rel_gate)
        # e_txt_embed = e_txt_embed_ori @ self.txt_W
        r_txt_embed = self.txt_relation_embeddings(relation)
        r_txt_embed_cross = self.txt_relation_embeddings_cross(relation)
        # e_mm_embed, attn_f = self.fuse_e(e_embed, e_img_embed, e_txt_embed, e_img_embed_ori, e_txt_embed_ori)

        e_d_img_embed_ori, disen_d_img, atten_i_d = self.diffusion_visual_moe(e_d_img_embed_head, rel_gate)
        r_d_img_embed = self.d_img_relation_embeddings(relation)

        e_img_embed_cross, disen_img_cross, atten_is = self.visual_structure_moe(e_img_embed_head, e_embed_head, rel_gate)
        e_txt_embed_cross, disen_txt_cross, atten_ds = self.text_structure_moe(e_txt_embed_head, e_embed_head, rel_gate)
        
        
        # e_embed, e_img_embed_ori, e_txt_embed_ori, e_img_embed_cross, e_txt_embed_cross, e_d_img_embed_ori, bias = self.attention(e_embed, e_img_embed_ori, e_txt_embed_ori, e_img_embed_cross, e_txt_embed_cross, e_d_img_embed_ori)
        
        # result, bias = self.attention([e_embed, e_img_embed_ori, e_txt_embed_ori, e_d_img_embed_ori])
        # result, bias = self.attention([e_embed, e_img_embed_ori, e_txt_embed_ori, e_img_embed_cross, e_txt_embed_cross, e_d_img_embed_ori])
        # e_embed, e_img_embed_ori, e_txt_embed_ori, e_d_img_embed_ori, *rest = result
        # e_embed, e_img_embed_ori, e_txt_embed_ori, e_img_embed_cross, e_txt_embed_cross, e_d_img_embed_ori, *rest = result
        # e_embed_new, e_img_embed_ori_new, e_txt_embed_ori_new, *rest = result
        
        # print("e_d_img_embed_ori:", e_d_img_embed_ori)
        e_mm_embed, attn_f = self.fuse_e(e_embed, e_img_embed_ori, e_txt_embed_ori, e_img_embed_cross, e_txt_embed_cross, e_d_img_embed_ori)
        # e_mm_embed, attn_f = self.fuse_e(e_embed, e_img_embed_ori, e_txt_embed_ori, modal6_emb=e_d_img_embed_ori)
        # e_mm_embed, attn_f = self.fuse_e(e_embed_new, e_img_embed_ori_new, e_txt_embed_ori_new, e_img_embed_cross, e_txt_embed_cross, e_d_img_embed_ori)
        r_mm_embed, _ = self.fuse_r(r_embed, r_img_embed, r_txt_embed, r_img_embed_cross, r_txt_embed_cross, r_d_img_embed)
        # r_mm_embed, _ = self.fuse_r(r_embed, r_img_embed, r_txt_embed, modal6_emb=r_d_img_embed)
        
        pred_s = self.TuckER_S(e_embed, r_embed)
        pred_i = self.TuckER_I(e_img_embed_ori, r_img_embed)
        pred_d = self.TuckER_D(e_txt_embed_ori, r_txt_embed)
        pred_is = self.TuckER_IS(e_img_embed_cross, r_img_embed_cross)
        pred_ds = self.TuckER_DS(e_txt_embed_cross, r_txt_embed_cross)
        pred_mm = self.TuckER_MM(e_mm_embed, r_mm_embed)
        pred_d_i = self.TuckER_DiffusionI(e_d_img_embed_ori, r_d_img_embed)
        
        all_s, _, _ = self.structure_moe(self.entity_embeddings.weight)
        all_v_ori, _, _ = self.visual_moe(self.img_entity_embeddings.weight)
        # all_v = all_v_ori @ self.img_W
        all_t_ori, _, _ = self.text_moe(self.txt_entity_embeddings.weight)
        # all_t = all_t_ori @ self.txt_W
        # all_f, _ = self.fuse_e(all_s, all_v, all_t, all_v_ori, all_t_ori)
        all_v_cross, _, _ = self.visual_structure_moe(self.img_entity_embeddings.weight, self.entity_embeddings.weight)
        all_t_cross, _, _ = self.text_structure_moe(self.txt_entity_embeddings.weight, self.entity_embeddings.weight)
        all_d_v_ori, _, _ = self.diffusion_visual_moe(self.d_img_entity_embeddings.weight)
        # result, _ = self.attention([all_s, all_v_ori, all_t_ori, all_d_v_ori])
        # result, _ = self.attention([all_s, all_v_ori, all_t_ori, all_v_cross, all_t_cross, all_d_v_ori])
        # all_s, all_v_ori, all_t_ori, all_d_v_ori, *rest = result
        # all_s, all_v_ori, all_t_ori, all_v_cross, all_t_cross, all_d_v_ori, *rest = result
        # all_f, _ = self.fuse_e(all_s, all_v_ori, all_t_ori, all_v_cross, all_t_cross, all_d_v_ori)
        all_f, _ = self.fuse_e(all_s, all_v_ori, all_t_ori, all_v_cross, all_t_cross, all_d_v_ori)
        # all_f, _ = self.fuse_e(all_s, all_v_ori, all_t_ori, all_v_cross, all_t_cross)

        contrastive_loss = 0
        # for output_i in [pred_s, pred_i, pred_d, pred_mm, pred_ds, pred_is]:
        #     logits = torch.matmul(output_i, output_i.t())
        #     labels = torch.arange(logits.size(0)).to(logits.device)
        #     contrastive_loss += F.cross_entropy(logits, labels)

        pred_s = torch.mm(pred_s, all_s.transpose(1, 0))
        # pred_i = torch.mm(pred_i, all_v.transpose(1, 0))
        # pred_d = torch.mm(pred_d, all_t.transpose(1, 0))
        pred_i = torch.mm(pred_i, all_v_ori.transpose(1, 0))
        pred_d = torch.mm(pred_d, all_t_ori.transpose(1, 0))
        pred_is = torch.mm(pred_is, all_v_cross.transpose(1, 0))
        pred_ds = torch.mm(pred_ds, all_t_cross.transpose(1, 0))
        pred_mm = torch.mm(pred_mm, all_f.transpose(1, 0))
        pred_d_i = torch.mm(pred_d_i, all_d_v_ori.transpose(1, 0))

        pred_s = torch.sigmoid(pred_s)
        pred_i = torch.sigmoid(pred_i)
        pred_d = torch.sigmoid(pred_d)
        pred_is = torch.sigmoid(pred_is)
        pred_ds = torch.sigmoid(pred_ds)
        pred_mm = torch.sigmoid(pred_mm)
        pred_d_i = torch.sigmoid(pred_d_i)
        if not self.training:
            return [pred_s, pred_i, pred_d, pred_is, pred_ds, pred_d_i, pred_mm], [atten_s, atten_i, atten_t, attn_f, atten_i_d, atten_is, atten_ds], [e_embed, e_img_embed_ori, e_txt_embed_ori], None
        else:
            # e_img_embed_norm = F.normalize(e_img_embed, p=2, dim=1)
            # e_txt_embed_norm = F.normalize(e_txt_embed, p=2, dim=1)
            # logit_scale = self.logit_scale.exp()
            # logits = logit_scale * e_img_embed_norm @ e_txt_embed_norm.T
            logits = None
            # return [pred_s, pred_i, pred_d, pred_is, pred_ds, pred_d_i, pred_mm], [disen_str, disen_img, disen_txt], [None, None, disen_d_img], None
            return [pred_s, pred_i, pred_d, pred_is, pred_ds, pred_d_i, pred_mm], [disen_str, disen_img, disen_txt], [disen_img_cross, disen_txt_cross, disen_d_img], None
    
    def get_batch_embeddings(self, batch_inputs):
        head = batch_inputs[:, 0]
        embed, disen_str, _ = self.structure_moe(self.entity_embeddings(head))
        img_embed, disen_img, _ = self.visual_moe(self.img_entity_embeddings(head))
        txt_embed, disen_txt, _ = self.text_moe(self.txt_entity_embeddings(head))
        _, disen_d_img, _ = self.diffusion_visual_moe(self.d_img_entity_embeddings(head))
        _, disen_img_cross, _ = self.visual_structure_moe(self.img_entity_embeddings(head), self.entity_embeddings(head))
        _, disen_txt_cross, _ = self.text_structure_moe(self.txt_entity_embeddings(head), self.entity_embeddings(head))
        # return [disen_str, disen_img, disen_txt], [None, None, disen_d_img]
        return [disen_str, disen_img, disen_txt], [disen_img_cross, disen_txt_cross, disen_d_img]


    def loss_func(self, output, target, weights=None, extra=False):
        bceloss = 0
        # if not extra:
        #     output = [output[0], output[1], output[2], output[5], output[6]]
        if weights is not None:
            # print(weights.shape)
            weights = torch.mean(weights, dim=0)
            # print(weights.shape)
            weights = torch.cat([weights, weights.new_tensor([1.0])]).detach()
            for i, output_i in enumerate(output):
                bceloss += weights[i] * self.bceloss(output_i, target)
        else:
            for output_i in output:
                bceloss += self.bceloss(output_i, target)
        contrastive_loss = 0
        # for output_i in output:
        #     logits = torch.matmul(output_i, output_i.t())
        #     labels = torch.arange(logits.size(0)).to(logits.device)
        #     contrastive_loss += F.cross_entropy(logits, labels)
        return bceloss + contrastive_loss

    def get_fusion_weights(self, batch_inputs):
        head = batch_inputs[:, 0]
        e_embed_head = self.entity_embeddings(head)
        e_img_embed_head = self.img_entity_embeddings(head)
        e_txt_embed_head = self.txt_entity_embeddings(head)
        e_d_img_embed_head = self.d_img_entity_embeddings(head)

        e_embed, _, _ = self.structure_moe(e_embed_head)
        e_img_embed_ori, _, _ = self.visual_moe(e_img_embed_head)
        e_txt_embed_ori, _, _ = self.text_moe(e_txt_embed_head)

        e_img_embed_cross, _, _ = self.visual_structure_moe(e_img_embed_head, e_embed_head)
        e_txt_embed_cross, _, _ = self.text_structure_moe(e_txt_embed_head, e_embed_head)

        e_d_img_embed_ori, _, _ = self.diffusion_visual_moe(e_d_img_embed_head)

        _, attn_f = self.fuse_e(e_embed, e_img_embed_ori, e_txt_embed_ori, e_img_embed_cross, e_txt_embed_cross, e_d_img_embed_ori)
        return torch.mean(attn_f, dim=0)

    def freeze_experts(self):
        for param in self.rel_gate.parameters():
            param.requires_grad = False

        for param in self.entity_embeddings.parameters():
            param.requires_grad = False
        for param in self.img_entity_embeddings.parameters():
            param.requires_grad = False
        for param in self.txt_entity_embeddings.parameters():
            param.requires_grad = False
        for param in self.d_img_entity_embeddings.parameters():
            param.requires_grad = False

        for param in self.structure_moe.parameters():
            param.requires_grad = False
        for param in self.relation_embeddings.parameters():
            param.requires_grad = False

        for param in self.visual_moe.parameters():
            param.requires_grad = False
        for param in self.img_relation_embeddings.parameters():
            param.requires_grad = False
        for param in self.img_relation_embeddings_cross.parameters():
            param.requires_grad = False

        for param in self.text_moe.parameters():
            param.requires_grad = False
        for param in self.txt_relation_embeddings.parameters():
            param.requires_grad = False
        for param in self.txt_relation_embeddings_cross.parameters():
            param.requires_grad = False

        for param in self.diffusion_visual_moe.parameters():
            param.requires_grad = False
        for param in self.d_img_relation_embeddings.parameters():
            param.requires_grad = False

        for param in self.visual_structure_moe.parameters():
            param.requires_grad = False
        for param in self.text_structure_moe.parameters():
            param.requires_grad = False
            
        for param in self.attention.parameters():
            param.requires_grad = False
        
        return