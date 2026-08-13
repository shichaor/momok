import torch
import numpy as np
from collections import defaultdict
from cca_zoo.nonparametric import KCCA
from cca_zoo.linear import CCA
import scipy


class Corpus:
    def __init__(self, args, train_data, val_data, test_data, entity2id, relation2id):
        self.device = args.device
        self.train_triples = train_data[0]
        self.val_triples = val_data[0]
        self.test_triples = test_data[0]
        self.max_batch_num = 1

        adj_indices = torch.LongTensor([train_data[1][0], train_data[1][1]])
        adj_values = torch.LongTensor([train_data[1][2]])
        self.train_adj_matrix = (adj_indices, adj_values)

        self.entity2id = {k: v for k, v in entity2id.items()}
        self.id2entity = {v: k for k, v in entity2id.items()}
        self.relation2id = {k: v for k, v in relation2id.items()}
        self.id2relation = {v: k for k, v in relation2id.items()}
        self.batch_size = args.batch_size

    def shuffle(self):
        raise NotImplementedError

    def get_batch(self, batch_num):
        raise NotImplementedError

    def get_validation_pred(self, model, split='test'):
        raise NotImplementedError


class ConvECorpus(Corpus):
    def __init__(self, args, train_data, val_data, test_data, entity2id, relation2id):
        super(ConvECorpus, self).__init__(args, train_data, val_data, test_data, entity2id, relation2id)
        rel_num = len(relation2id)
        for k, v in relation2id.items():
            self.relation2id[k+'_reverse'] = v+rel_num
        self.id2relation = {v: k for k, v in self.relation2id.items()}

        sr2o = {}
        for (head, relation, tail) in self.train_triples:
            if (head, relation) not in sr2o.keys():
                sr2o[(head, relation)] = set()
            if (tail, relation+rel_num) not in sr2o.keys():
                sr2o[(tail, relation+rel_num)] = set()
            sr2o[(head, relation)].add(tail)
            sr2o[(tail, relation+rel_num)].add(head)

        self.triples = {}
        self.train_indices = [{'triple': (head, relation, -1), 'label': list(sr2o[(head, relation)])}
                              for (head, relation), tail in sr2o.items()]
        self.triples['train'] = [{'triple': (head, relation, -1), 'label': list(sr2o[(head, relation)])}
                                 for (head, relation), tail in sr2o.items()]

        if len(self.train_indices) % self.batch_size == 0:
            self.max_batch_num = len(self.train_indices) // self.batch_size
        else:
            self.max_batch_num = len(self.train_indices) // self.batch_size + 1

        for (head, relation, tail) in self.val_triples:
            if (head, relation) not in sr2o.keys():
                sr2o[(head, relation)] = set()
            if (tail, relation+rel_num) not in sr2o.keys():
                sr2o[(tail, relation+rel_num)] = set()
            sr2o[(head, relation)].add(tail)
            sr2o[(tail, relation+rel_num)].add(head)

        for (head, relation, tail) in self.test_triples:
            if (head, relation) not in sr2o.keys():
                sr2o[(head, relation)] = set()
            if (tail, relation+rel_num) not in sr2o.keys():
                sr2o[(tail, relation+rel_num)] = set()
            sr2o[(head, relation)].add(tail)
            sr2o[(tail, relation+rel_num)].add(head)

        self.val_head_indices = [{'triple': (tail, relation + rel_num, head), 'label': list(sr2o[(tail, relation + rel_num)])}
                                 for (head, relation, tail) in self.val_triples]
        self.val_tail_indices = [{'triple': (head, relation, tail), 'label': list(sr2o[(head, relation)])}
                                 for (head, relation, tail) in self.val_triples]
        self.test_head_indices = [{'triple': (tail, relation + rel_num, head), 'label': list(sr2o[(tail, relation + rel_num)])}
                                 for (head, relation, tail) in self.test_triples]
        self.test_tail_indices = [{'triple': (head, relation, tail), 'label': list(sr2o[(head, relation)])}
                                 for (head, relation, tail) in self.test_triples]


    def read_batch(self, batch):
        triple, label = [_.to(self.device) for _ in batch]
        return triple, label

    def shuffle(self):
        np.random.shuffle(self.train_indices)

    def get_batch(self, batch_num, device="cpu"):
        if (batch_num + 1) * self.batch_size <= len(self.train_indices):
            batch = self.train_indices[batch_num * self.batch_size: (batch_num+1) * self.batch_size]
        else:
            batch = self.train_indices[batch_num * self.batch_size:]
        batch_indices = torch.LongTensor([indice['triple'] for indice in batch]).to(device)
        label = [np.int32(indice['label']) for indice in batch]
        y = torch.zeros((len(batch), len(self.entity2id)), dtype=torch.float32, device=device)
        if label:
        # 展平索引：记录每个正标签对应的 [行号, 列号]
            rows = []
            cols = []
            for i, pos_list in enumerate(label):
                # 如果 pos_list 不为空
                if pos_list.any():
                    rows.extend([i] * len(pos_list))
                    cols.extend(pos_list)
            
            if rows:
                # 一次性将对应位置置为 1.0（极速，由 CUDA 核函数完成）
                y[rows, cols] = 1.0
        # for idx in range(len(label)):
        #     for l in label[idx]:
        #         y[idx][l] = 1.0
        # import time
        # t_start = time.perf_counter()
        # print(y.shape)
        y = 0.9 * y + (1.0 / len(self.entity2id))
        # t_end = time.perf_counter()
        # print(f"耗时: {t_end - t_start:.4f} 秒", flush=True)  # 正常应小于 0.01 秒
        # batch_values = torch.FloatTensor(y)

        '''index = []
        for idx in range(len(label)):
            pos = label[idx]
            np.random.shuffle(pos)
            neg = np.int32(list(range((len(self.entity2id)))))
            np.random.shuffle(neg)
            if len(pos) >= 10:
                index.append(np.concatenate((pos[:10], neg[:90])))
            else:
                index.append(np.concatenate((pos, neg[:100-len(pos)])))
        y = torch.FloatTensor(y)
        index = torch.LongTensor(index)
        batch_values = torch.gather(y, dim=1, index=index)'''

        return batch_indices, y#, index

    def get_validation_pred(self, model, split='test', round=0, device="cpu"):
        ranks_head, ranks_tail = [], []
        branch_ranks_head, branch_ranks_tail = {}, {}
        for i in range(7):
            branch_ranks_head[i] = []
            branch_ranks_tail[i] = []
        target_ranks_head, target_ranks_tail = {}, {}
        for i in range(7):
            target_ranks_head[i] = []
            target_ranks_tail[i] = []
        max_ranks_head, max_ranks_tail = {}, {}
        for i in range(7):
            max_ranks_head[i] = []
            max_ranks_tail[i] = []
        reciprocal_ranks_head, reciprocal_ranks_tail = [], []
        hits_at_100_head, hits_at_100_tail = 0, 0
        hits_at_10_head, hits_at_10_tail = 0, 0
        hits_at_3_head, hits_at_3_tail = 0, 0
        hits_at_1_head, hits_at_1_tail = 0, 0

        rel_pred_dict = defaultdict(list)
        att_s = []
        att_i = []
        att_t = []
        att_mm = []
        att_i_d = []
        att_is = []
        att_ds = []
        kcca = KCCA(latent_dimensions=1)

        def _branch_metrics(scores, y, target):
            b_range = torch.arange(scores.shape[0], device=scores.device)
            target_score = scores[b_range, target]          # 先把正例分抠出来
            max_score = scores.max(dim=1)[0]                # 每样本最高分数 (batch_size,)
            scores = torch.where(y.bool(),                   # 对应旧代码 pred[b_range,target]=target_pred
                                torch.zeros_like(scores), scores)
            scores[b_range, target] = target_score
            scores = scores.cpu().numpy()
            ranks = []
            for i in range(scores.shape[0]):
                scr = scores[i]
                tar = target[i]
                tar_scr = scr[tar]
                scr = np.delete(scr, tar)                   # 挖掉正例
                rand = np.random.randint(scr.shape[0])      # 随机插入位置
                scr = np.insert(scr, rand, tar_scr)
                sorted_idx = np.argsort(-scr, kind='stable')
                ranks.append(np.where(sorted_idx == rand)[0][0] + 1)
            return ranks, target_score, max_score
        
        def _late_fusion(pred_list, weights=None, strategy=['mean']):
            alpha = 1
            beta = 1
            gamma = 1
            if weights is None:
                weights = torch.zeros((len(pred_list), pred_list[0].shape[0]), device=device)
                if 'mean' in strategy:
                    weights += [1.0 / len(pred_list)] * pred_list[0].shape[0]
                if 'confidence' in strategy:
                    top1 = [torch.max(p, dim=1)[0] for p in pred_list]
                    top2 = [torch.topk(p, k=2, dim=1)[0][:, 1] for p in pred_list]
                    margins = torch.tensor([t1 - t2 for t1, t2 in zip(top1, top2)], device=device)
                    entropy = torch.tensor([-(p * torch.log(p + 1e-10)).sum(dim=1) for p in pred_list], device=device)
                    std = torch.tensor([torch.std(p, dim=1) for p in pred_list], device=device)
                    confidence_score = margins
                    weights += alpha * confidence_score
                if 'redundancy' in strategy:
                    redundancy = []
                    check_matrix = torch.tensor([[0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0],
                                                 [1, 1, 0, 0, 0, 0, 0],
                                                 [1, 0, 1, 0, 0, 0, 0],
                                                 [0, 1, 1, 0, 0, 0, 0],
                                                 [0, 0, 0, 0, 0, 0, 0]], device=device) # m[i,j]=1 means i may be redundant to j
                    for modal_i in range(check_matrix.shape[0]):
                        redundant_score = torch.zeros(pred_list[0].shape[0], device=device)
                        for modal_j in range(check_matrix.shape[1]):
                            if check_matrix[modal_i, modal_j] == 1:
                                rank_i = torch.argsort(pred_list[modal_i], dim=1)
                                rank_j = torch.argsort(pred_list[modal_j], dim=1)
                                corr = [scipy.stats.spearmanr(j.cpu().numpy(), i.cpu().numpy())[0] for i, j in zip(rank_i, rank_j)]
                                redundant_score = torch.max(redundant_score, torch.tensor(corr, device=device))
                        redundancy.append(redundant_score)
                    redundancy = torch.tensor(redundancy, device=device)
                    weights += beta * (-redundancy)
                if 'conflict' in strategy:
                    assert 'confidence' in strategy, "Conflict strategy requires confidence strategy to be enabled."
                    K = 30
                    conflict = []
                    ranks = [torch.argsort(p, dim=1) for p in pred_list]
                    for i in range(len(pred_list)):
                        conflict_score = torch.zeros(pred_list[0].shape[0], device=device)
                        rank_i = ranks[i]
                        topk_i = rank_i[:, :K]
                        for j in range(len(pred_list)):
                            if i != j:
                                rank_j = ranks[j]
                                topk_j = rank_j[:, :K]
                                overlap = torch.tensor([len(set(topk_i[k].cpu().numpy()).intersection(set(topk_j[k].cpu().numpy()))) for k in range(topk_i.shape[0])], device=device)
                                conflict_score += torch.mean(confidence_score * (1.0 - overlap.float() / K))
                        conflict.append(conflict_score)
                    weights += gamma * (-conflict)
            weights = torch.tensor(weights, device=device)
            weights = torch.softmax(weights, dim=0)
            pred = sum([w * p for w, p in zip(weights, pred_list)])
            return pred

        if split == 'val':
            head_indices = self.val_head_indices
            tail_indices = self.val_tail_indices
        else:
            head_indices = self.test_head_indices
            tail_indices = self.test_tail_indices

        if len(head_indices) % self.batch_size == 0:
            max_batch_num = len(head_indices) // self.batch_size
        else:
            max_batch_num = len(head_indices) // self.batch_size + 1

        emb_i_head = []
        emb_t_head = []
        emb_s_head = []
        extra_emb_head = []
        emb_i_tail = []
        emb_t_tail = []
        emb_s_tail = []
        extra_emb_tail = []
    
        for batch_num in range(max_batch_num):
            if (batch_num + 1) * self.batch_size <= len(head_indices):
                head_batch = head_indices[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]
                tail_batch = tail_indices[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]
            else:
                head_batch = head_indices[batch_num * self.batch_size:]
                tail_batch = tail_indices[batch_num * self.batch_size:]

            head_batch_indices = torch.LongTensor([indice['triple'] for indice in head_batch])
            head_batch_indices = head_batch_indices.to(self.device)
            rel_ids = head_batch_indices[:, 1]
            pred, attention, emb, extra_emb = model.forward(head_batch_indices)
            if extra_emb is not None:
                emb_s, emb_i, emb_t = emb
                emb_s_head.append(emb_s)
                emb_i_head.append(emb_i)
                emb_t_head.append(emb_t)
                extra_emb_head.append(extra_emb)
                # c1, c2 = kcca.fit_transform([emb_s.cpu().numpy(), emb_i.cpu().numpy()])
                # score_si = torch.cosine_similarity(torch.tensor(c1).to(self.device), torch.tensor(c2).to(self.device))
                # c1, c2 = kcca.fit_transform([emb_s.cpu().numpy(), emb_t.cpu().numpy()])
                # score_st = torch.cosine_similarity(torch.tensor(c1).to(self.device), torch.tensor(c2).to(self.device))
                # c1, c2 = kcca.fit_transform([emb_i.cpu().numpy(), emb_t.cpu().numpy()])
                # score_it = torch.cosine_similarity(torch.tensor(c1).to(self.device), torch.tensor(c2).to(self.device))
                # c1, c2 = kcca.fit_transform([emb_s.cpu().numpy(), extra_emb.cpu().numpy()])
                # score_s_extra = torch.cosine_similarity(torch.tensor(c1).to(self.device), torch.tensor(c2).to(self.device))
                # c1, c2 = kcca.fit_transform([emb_i.cpu().numpy(), extra_emb.cpu().numpy()])
                # score_i_extra = torch.cosine_similarity(torch.tensor(c1).to(self.device), torch.tensor(c2).to(self.device))
                # c1, c2 = kcca.fit_transform([emb_t.cpu().numpy(), extra_emb.cpu().numpy()])
                # score_t_extra = torch.cosine_similarity(torch.tensor(c1).to(self.device), torch.tensor(c2).to(self.device))
                # print("similarity of original embeddings: score_si:{:.4f}, score_st:{:.4f}, score_it:{:.4f}".format(score_si.mean().item(), score_st.mean().item(), score_it.mean().item()))
                # print("similarity of original embeddings and extra embedding: score_s_extra:{:.4f}, score_i_extra:{:.4f}, score_t_extra:{:.4f}".format(score_s_extra.mean().item(), score_i_extra.mean().item(), score_t_extra.mean().item()))

            for i in range(pred[0].shape[0]):
                h, r, t = head_batch_indices[i][0].item(), head_batch_indices[i][1].item(), head_batch_indices[i][2].item()
                atts = attention[0][i]
                atti = attention[1][i]
                attt = attention[2][i]
                attmm = attention[3][i]
                atti_d = attention[4][i]
                attis = attention[5][i]
                attds = attention[6][i]
                att_s.append((h, r, t, atts))
                att_i.append((h, r, t, atti))
                att_t.append((h, r, t, attt))
                att_mm.append((h, r, t, attmm))
                att_i_d.append((h, r, t, atti_d))
                att_is.append((h, r, t, attis))
                att_ds.append((h, r, t, attds))
            # weights = torch.mean(torch.stack([a[-1] for a in att_mm]), dim=0)
            # weights = torch.cat([weights, weights.new_tensor([1.0])])
            # pred = sum([w * p for w, p in zip(weights, pred)])
            raw_pred_head = pred
            # raw_pred_head = [pred[0], pred[1], pred[2], pred[5], pred[6]]
            pred = _late_fusion(raw_pred_head)
            label = [np.int32(indice['label']) for indice in head_batch]
            y = np.zeros((len(head_batch), len(self.entity2id)), dtype=np.float32)
            for idx in range(len(label)):
                for l in label[idx]:
                    y[idx][l] = 1.0
            y = torch.FloatTensor(y).to(self.device)
            target = head_batch_indices[:, 2]
            b_range = torch.arange(pred.shape[0], device=self.device)
            target_pred = pred[b_range, target]
            pred = torch.where(y.bool(), torch.zeros_like(pred), pred)
            pred[b_range, target] = target_pred
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
            # pred.shape[0] = batch_size
            for i in range(pred.shape[0]):
                scores = pred[i]
                tar = target[i]
                tar_scr = scores[tar]
                scores = np.delete(scores, tar)
                rand = np.random.randint(scores.shape[0])
                scores = np.insert(scores, rand, tar_scr)
                sorted_indices = np.argsort(-scores, kind='stable')
                # higher is better
                ranks_head.append(np.where(sorted_indices == rand)[0][0]+1)
                reciprocal_ranks_head.append(1.0 / ranks_head[-1])
                rel_pred_dict[rel_ids[i].item()].append(ranks_head[-1])
            for pred_i, i in zip(raw_pred_head, range(len(raw_pred_head))):
                branch_ranks, target_score, max_score = _branch_metrics(pred_i, y, target)
                branch_ranks_head[i].extend(branch_ranks)
                target_ranks_head[i].extend(target_score.cpu().numpy())
                max_ranks_head[i].extend(max_score.cpu().numpy())

            tail_batch_indices = torch.LongTensor([indice['triple'] for indice in tail_batch])
            tail_batch_indices = tail_batch_indices.to(self.device)
            rel_ids = tail_batch_indices[:, 1]
            pred, attention, emb, extra_emb = model.forward(tail_batch_indices)
            if extra_emb is not None:
                emb_s, emb_i, emb_t = emb
                emb_s_tail.append(emb_s)
                emb_i_tail.append(emb_i)
                emb_t_tail.append(emb_t)
                extra_emb_tail.append(extra_emb)
                # c1, c2 = kcca.fit_transform([emb_s.cpu().numpy(), emb_i.cpu().numpy()])
                # score_si = torch.cosine_similarity(torch.tensor(c1).to(self.device), torch.tensor(c2).to(self.device))
                # c1, c2 = kcca.fit_transform([emb_s.cpu().numpy(), emb_t.cpu().numpy()])
                # score_st = torch.cosine_similarity(torch.tensor(c1).to(self.device), torch.tensor(c2).to(self.device))
                # c1, c2 = kcca.fit_transform([emb_i.cpu().numpy(), emb_t.cpu().numpy()])
                # score_it = torch.cosine_similarity(torch.tensor(c1).to(self.device), torch.tensor(c2).to(self.device))
                # c1, c2 = kcca.fit_transform([emb_s.cpu().numpy(), extra_emb.cpu().numpy()])
                # score_s_extra = torch.cosine_similarity(torch.tensor(c1).to(self.device), torch.tensor(c2).to(self.device))
                # c1, c2 = kcca.fit_transform([emb_i.cpu().numpy(), extra_emb.cpu().numpy()])
                # score_i_extra = torch.cosine_similarity(torch.tensor(c1).to(self.device), torch.tensor(c2).to(self.device))
                # c1, c2 = kcca.fit_transform([emb_t.cpu().numpy(), extra_emb.cpu().numpy()])
                # score_t_extra = torch.cosine_similarity(torch.tensor(c1).to(self.device), torch.tensor(c2).to(self.device))
                # print("similarity of original embeddings: score_si:{:.4f}, score_st:{:.4f}, score_it:{:.4f}".format(score_si.mean().item(), score_st.mean().item(), score_it.mean().item()))
                # print("similarity of original embeddings and extra embedding: score_s_extra:{:.4f}, score_i_extra:{:.4f}, score_t_extra:{:.4f}".format(score_s_extra.mean().item(), score_i_extra.mean().item(), score_t_extra.mean().item()))
            
            for i in range(pred[0].shape[0]):
                h, r, t = tail_batch_indices[i][0].item(), tail_batch_indices[i][1].item(), tail_batch_indices[i][2].item()
                atts = attention[0][i]
                atti = attention[1][i]
                attt = attention[2][i]
                attmm = attention[3][i]
                atti_d = attention[4][i]
                attis = attention[5][i]
                attds = attention[6][i]
                att_s.append((h, r, t, atts))
                att_i.append((h, r, t, atti))
                att_t.append((h, r, t, attt))
                att_mm.append((h, r, t, attmm))
                att_i_d.append((h, r, t, atti_d))
                att_is.append((h, r, t, attis))
                att_ds.append((h, r, t, attds))
            # weights = torch.mean(torch.stack([a[-1] for a in att_mm]), dim=0)
            # weights = torch.cat([weights, weights.new_tensor([1.0])])
            # pred = sum([w * p for w, p in zip(weights, pred)])
            raw_pred_tail = pred
            # raw_pred_tail = [pred[0], pred[1], pred[2], pred[5], pred[6]]
            pred = _late_fusion(raw_pred_tail)
            label = [np.int32(indice['label']) for indice in tail_batch]
            y = np.zeros((len(tail_batch), len(self.entity2id)), dtype=np.float32)
            for idx in range(len(label)):
                for l in label[idx]:
                    y[idx][l] = 1.0
            y = torch.FloatTensor(y).to(self.device)
            target = tail_batch_indices[:, 2]
            b_range = torch.arange(pred.shape[0], device=self.device)
            target_pred = pred[b_range, target]
            pred = torch.where(y.bool(), torch.zeros_like(pred), pred)
            pred[b_range, target] = target_pred
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
            for i in range(pred.shape[0]):
                scores = pred[i]
                tar = target[i]
                tar_scr = scores[tar]
                scores = np.delete(scores, tar)
                rand = np.random.randint(scores.shape[0])
                scores = np.insert(scores, rand, tar_scr)
                sorted_indices = np.argsort(-scores, kind='stable')
                ranks_tail.append(np.where(sorted_indices == rand)[0][0] + 1)
                reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
                rel_pred_dict[rel_ids[i].item()].append(ranks_head[-1])
            for pred_i, i in zip(raw_pred_tail, range(len(raw_pred_tail))):
                branch_ranks, target_score, max_score = _branch_metrics(pred_i, y, target)
                branch_ranks_tail[i].extend(branch_ranks)
                target_ranks_tail[i].extend(target_score.cpu().numpy())
                max_ranks_tail[i].extend(max_score.cpu().numpy())
        


        def _compute_branch_metrics(branch_ranks):
            branch_metrics = {}
            for i in range(len(branch_ranks)):
                ranks = branch_ranks[i]
                if len(ranks) == 0:
                    continue
                hits_at_100, hits_at_10, hits_at_3, hits_at_1 = 0, 0, 0, 0
                for r in ranks:
                    if r <= 100:
                        hits_at_100 += 1
                    if r <= 10:
                        hits_at_10 += 1
                    if r <= 3:
                        hits_at_3 += 1
                    if r == 1:
                        hits_at_1 += 1
                branch_metrics[i] = {
                    "Hits@100": hits_at_100 / len(ranks),
                    "Hits@10": hits_at_10 / len(ranks),
                    "Hits@3": hits_at_3 / len(ranks),
                    "Hits@1": hits_at_1 / len(ranks),
                    "MR": sum(ranks) / len(ranks),
                    "MRR": sum([1.0 / r for r in ranks]) / len(ranks)
                }
            return branch_metrics
        
        import matplotlib.pyplot as plt
        def plot_score_distribution(branch_scores, prefix, titles):
            for i in range(len(branch_scores)):
                title = titles[i]
                scores = branch_scores[i]
                plt.figure(figsize=(10, 6))
                plt.hist(scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
                plt.title(f'{prefix} of {title}')
                plt.xlabel('Score')
                plt.ylabel('Frequency')
                plt.grid(axis='y', alpha=0.75)
                plt.savefig(f'figs/{prefix}_{title}_{round}.png')
        
        def compute_mcca_metrics(views_gpu, target_idx=0, n_components=128, reg=0.01):
            """
            直接接受GPU上的torch.Tensor作为输入，全程在显存中计算
            参数:
                views_gpu: list of torch.Tensor (已在GPU上), 形状均为 (n_samples, dim)
                target_idx: 目标模态索引 (0, 1, 2, 3)
                n_components: 统一语义空间维度 (建议 ≤ 最小原始维度)
                reg: 正则化系数
            返回:
                gain: 边际信息增益 (float)
                uniqueness: 独特率 (float, 越高越不可替代)
                elapsed: 计算耗时 (秒)
            """
            # start = time.perf_counter()
            
            # ---------- 1. 关键安全操作 ----------
            # 必须 detach() 切断梯度，否则计算图会累积导致显存泄漏
            # 同时转为双精度 (float64) 保证特征分解的数值稳定性，若显存紧张可改为 .float()
            views = [X.detach().double() for X in views_gpu]  
            
            # ---------- 核心GPU辅助函数：返回互信息和投影 ----------
            def _mcca_gpu(views_sub, comp, reg_val, need_proj=True):
                # 标准化 (Z-score)
                views_std = []
                for X in views_sub:
                    mean = X.mean(dim=0, keepdim=True)
                    std = X.std(dim=0, keepdim=True) + 1e-10
                    views_std.append((X - mean) / std)
                
                # 白化: D^{-1/2}
                D_inv_sqrt_list = []
                for X in views_std:
                    cov = X.T @ X + reg_val * torch.eye(X.shape[1], device=device, dtype=torch.float64)
                    evals, evecs = torch.linalg.eigh(cov)
                    evals = torch.clamp(evals, min=1e-10)
                    inv_sqrt = evecs @ torch.diag(1.0 / torch.sqrt(evals)) @ evecs.T
                    D_inv_sqrt_list.append(inv_sqrt)
                
                # 构建全协方差矩阵 R = D^{-1/2} * C * D^{-1/2}
                D_inv_sqrt = torch.block_diag(*D_inv_sqrt_list)
                X_all = torch.cat(views_std, dim=1)
                C = X_all.T @ X_all
                R = D_inv_sqrt @ C @ D_inv_sqrt
                
                # 特征分解，计算互信息
                evals_all, evecs_all = torch.linalg.eigh(R)
                top_evals = torch.topk(evals_all, k=comp)[0]
                rho_sq = torch.clamp(top_evals, min=0, max=0.9999)
                I_total = -0.5 * torch.sum(torch.log(1 - rho_sq))
                
                if need_proj:
                    # 取最大的 comp 个特征向量 (evals_all是升序)
                    idx = torch.argsort(evals_all)[-comp:]
                    V = evecs_all[:, idx]
                    W = D_inv_sqrt @ V
                    
                    # 按原始维度分割 W
                    dims = [X.shape[1] for X in views_std]
                    Ws = torch.split(W, split_size_or_sections=dims, dim=0)
                    
                    # 计算投影
                    projections = [X @ Ws[i] for i, X in enumerate(views_std)]
                    return I_total, projections
                else:
                    return I_total, None

            # ---------- 2. 计算增益 ----------
            # 全量4组
            I_all, proj_all = _mcca_gpu(views, n_components, reg, need_proj=True)
            # 排除目标模态
            views_rest = [views[i] for i in range(len(views)) if i != target_idx]
            I_rest, _ = _mcca_gpu(views_rest, n_components, reg, need_proj=False)
            gain = (I_all - I_rest).item()

            # ---------- 3. 计算独特率 (基于投影) ----------
            Y = proj_all[target_idx]
            X_list = [proj_all[i] for i in range(len(proj_all)) if i != target_idx]
            X_other = torch.cat(X_list, dim=1)
            
            # 添加偏置列
            ones = torch.ones((X_other.shape[0], 1), device=device, dtype=torch.float64)
            X_aug = torch.cat([X_other, ones], dim=1)
            
            # 最小二乘解析解 + 正则化
            XTX = X_aug.T @ X_aug + reg * torch.eye(X_aug.shape[1], device=device, dtype=torch.float64)
            XTY = X_aug.T @ Y
            beta = torch.linalg.solve(XTX, XTY)
            
            Y_pred = X_aug @ beta
            SS_res = torch.sum((Y - Y_pred) ** 2)
            SS_tot = torch.sum((Y - Y.mean(dim=0, keepdim=True)) ** 2)
            r2 = (1 - SS_res / (SS_tot + 1e-10)).item()
            uniqueness = 1 - r2

            # elapsed = time.perf_counter() - start
            return gain, uniqueness
        
        branch_metrics_head = _compute_branch_metrics(branch_ranks_head)
        branch_metrics_tail = _compute_branch_metrics(branch_ranks_tail)
        if len(emb_s_head) > 0:
            emb_s_head = torch.cat(emb_s_head, dim=0)
            emb_i_head = torch.cat(emb_i_head, dim=0)
            emb_t_head = torch.cat(emb_t_head, dim=0)
            extra_emb_head = torch.cat(extra_emb_head, dim=0)
            gain, uniqueness = compute_mcca_metrics([emb_s_head, emb_i_head, emb_t_head, extra_emb_head], target_idx=3, n_components=128, reg=0.01)
            print(f"Head Gain: {gain:.4f}, Uniqueness: {uniqueness:.4f}")
            gain, uniqueness = compute_mcca_metrics([emb_s_head, emb_i_head, emb_t_head, extra_emb_head], target_idx=0, n_components=128, reg=0.01)
            print(f"cmp Gain: {gain:.4f}, Uniqueness: {uniqueness:.4f}")
            emb_s_tail = torch.cat(emb_s_tail, dim=0)
            emb_i_tail = torch.cat(emb_i_tail, dim=0)
            emb_t_tail = torch.cat(emb_t_tail, dim=0)
            extra_emb_tail = torch.cat(extra_emb_tail, dim=0)
            gain, uniqueness = compute_mcca_metrics([emb_s_tail, emb_i_tail, emb_t_tail, extra_emb_tail], target_idx=3, n_components=128, reg=0.01)
            print(f"Tail Gain: {gain:.4f}, Uniqueness: {uniqueness:.4f}")
            gain, uniqueness = compute_mcca_metrics([emb_s_tail, emb_i_tail, emb_t_tail, extra_emb_tail], target_idx=0, n_components=128, reg=0.01)
            print(f"cmp Gain: {gain:.4f}, Uniqueness: {uniqueness:.4f}")
        branch_title = ['pred_s', 'pred_i', 'pred_d', 'pred_is', 'pred_ds', 'pred_d_i', 'pred_mm']
        # plot_score_distribution([target_ranks_head[i] for i in range(len(branch_title))], 'Target Scores Head', branch_title)
        # plot_score_distribution([max_ranks_head[i] for i in range(len(branch_title))], 'Max Scores Head', branch_title)
        # plot_score_distribution([target_ranks_tail[i] for i in range(len(branch_title))], 'Target Scores Tail', branch_title)
        # plot_score_distribution([max_ranks_tail[i] for i in range(len(branch_title))], 'Max Scores Tail', branch_title)
        # branch_title = ['pred_s', 'pred_i', 'pred_d', 'pred_d_i', 'pred_mm']
        for i, title in enumerate(branch_title):
            print(f"Branch {title} Head Metrics: {branch_metrics_head[i]}")
            print(f"Branch {title} Tail Metrics: {branch_metrics_tail[i]}")
        
        for i in range(len(ranks_head)):
            if ranks_head[i] <= 100:
                hits_at_100_head += 1
            if ranks_head[i] <= 10:
                hits_at_10_head += 1
            if ranks_head[i] <= 3:
                hits_at_3_head += 1
            if ranks_head[i] == 1:
                hits_at_1_head += 1

        for i in range(len(ranks_tail)):
            if ranks_tail[i] <= 100:
                hits_at_100_tail += 1
            if ranks_tail[i] <= 10:
                hits_at_10_tail += 1
            if ranks_tail[i] <= 3:
                hits_at_3_tail += 1
            if ranks_tail[i] == 1:
                hits_at_1_tail += 1

        assert len(ranks_head) == len(reciprocal_ranks_head)
        assert len(ranks_tail) == len(reciprocal_ranks_tail)

        hits_100_head = hits_at_100_head / len(ranks_head)
        hits_10_head = hits_at_10_head / len(ranks_head)
        hits_3_head = hits_at_3_head / len(ranks_head)
        hits_1_head = hits_at_1_head / len(ranks_head)
        mean_rank_head = sum(ranks_head) / len(ranks_head)
        mean_reciprocal_rank_head = sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)

        hits_100_tail = hits_at_100_tail / len(ranks_tail)
        hits_10_tail = hits_at_10_tail / len(ranks_tail)
        hits_3_tail = hits_at_3_tail / len(ranks_tail)
        hits_1_tail = hits_at_1_tail / len(ranks_tail)
        mean_rank_tail = sum(ranks_tail) / len(ranks_tail)
        mean_reciprocal_rank_tail = sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)

        hits_100 = (hits_at_100_head / len(ranks_head) + hits_at_100_tail / len(ranks_tail)) / 2
        hits_10 = (hits_at_10_head / len(ranks_head) + hits_at_10_tail / len(ranks_tail)) / 2
        hits_3 = (hits_at_3_head / len(ranks_head) + hits_at_3_tail / len(ranks_tail)) / 2
        hits_1 = (hits_at_1_head / len(ranks_head) + hits_at_1_tail / len(ranks_tail)) / 2
        mean_rank = (sum(ranks_head) / len(ranks_head) + sum(ranks_tail) / len(ranks_tail)) / 2
        mean_reciprocal_rank = (sum(reciprocal_ranks_head) / len(reciprocal_ranks_head) + sum(
            reciprocal_ranks_tail) / len(reciprocal_ranks_tail)) / 2

        metrics = {
            "Hits@100": hits_100,
            "Hits@10": hits_10,
            "Hits@3": hits_3,
            "Hits@1": hits_1,
            "MR": mean_rank,
            "MRR": mean_reciprocal_rank
        }

        return metrics, [att_s, att_i, att_t, att_mm, att_i_d, att_is, att_ds]
        # return metrics, [att_s, att_i, att_t, att_mm]
    
    def get_validation_pred_signle(self, model, split='test', index=0):
        ranks_head, ranks_tail = [], []
        reciprocal_ranks_head, reciprocal_ranks_tail = [], []
        hits_at_100_head, hits_at_100_tail = 0, 0
        hits_at_10_head, hits_at_10_tail = 0, 0
        hits_at_3_head, hits_at_3_tail = 0, 0
        hits_at_1_head, hits_at_1_tail = 0, 0
        rel_pred_dict = defaultdict(list)

        if split == 'val':
            head_indices = self.val_head_indices
            tail_indices = self.val_tail_indices
        else:
            head_indices = self.test_head_indices
            tail_indices = self.test_tail_indices

        if len(head_indices) % self.batch_size == 0:
            max_batch_num = len(head_indices) // self.batch_size
        else:
            max_batch_num = len(head_indices) // self.batch_size + 1
        for batch_num in range(max_batch_num):
            if (batch_num + 1) * self.batch_size <= len(head_indices):
                head_batch = head_indices[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]
                tail_batch = tail_indices[batch_num * self.batch_size: (batch_num + 1) * self.batch_size]
            else:
                head_batch = head_indices[batch_num * self.batch_size:]
                tail_batch = tail_indices[batch_num * self.batch_size:]

            head_batch_indices = torch.LongTensor([indice['triple'] for indice in head_batch])
            head_batch_indices = head_batch_indices.to(self.device)
            rel_ids = head_batch_indices[:, 1]
            pred, attention, _, _ = model.forward(head_batch_indices)
            pred = pred[index]
            label = [np.int32(indice['label']) for indice in head_batch]
            y = np.zeros((len(head_batch), len(self.entity2id)), dtype=np.float32)
            for idx in range(len(label)):
                for l in label[idx]:
                    y[idx][l] = 1.0
            y = torch.FloatTensor(y).to(self.device)
            target = head_batch_indices[:, 2]
            b_range = torch.arange(pred.shape[0], device=self.device)
            target_pred = pred[b_range, target]
            pred = torch.where(y.bool(), torch.zeros_like(pred), pred)
            pred[b_range, target] = target_pred
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
            for i in range(pred.shape[0]):
                scores = pred[i]
                tar = target[i]
                tar_scr = scores[tar]
                scores = np.delete(scores, tar)
                rand = np.random.randint(scores.shape[0])
                scores = np.insert(scores, rand, tar_scr)
                sorted_indices = np.argsort(-scores, kind='stable')
                # higher is better
                ranks_head.append(np.where(sorted_indices == rand)[0][0]+1)
                reciprocal_ranks_head.append(1.0 / ranks_head[-1])
                rel_pred_dict[rel_ids[i].item()].append(ranks_head[-1])

            tail_batch_indices = torch.LongTensor([indice['triple'] for indice in tail_batch])
            tail_batch_indices = tail_batch_indices.to(self.device)
            rel_ids = tail_batch_indices[:, 1]
            pred, attention, _, _ = model.forward(tail_batch_indices)
            pred = pred[index]
            label = [np.int32(indice['label']) for indice in tail_batch]
            y = np.zeros((len(tail_batch), len(self.entity2id)), dtype=np.float32)
            for idx in range(len(label)):
                for l in label[idx]:
                    y[idx][l] = 1.0
            y = torch.FloatTensor(y).to(self.device)
            target = tail_batch_indices[:, 2]
            b_range = torch.arange(pred.shape[0], device=self.device)
            target_pred = pred[b_range, target]
            pred = torch.where(y.bool(), torch.zeros_like(pred), pred)
            pred[b_range, target] = target_pred
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
            for i in range(pred.shape[0]):
                scores = pred[i]
                tar = target[i]
                tar_scr = scores[tar]
                scores = np.delete(scores, tar)
                rand = np.random.randint(scores.shape[0])
                scores = np.insert(scores, rand, tar_scr)
                sorted_indices = np.argsort(-scores, kind='stable')
                ranks_tail.append(np.where(sorted_indices == rand)[0][0] + 1)
                reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])
                rel_pred_dict[rel_ids[i].item()].append(ranks_head[-1])

        for i in range(len(ranks_head)):
            if ranks_head[i] <= 100:
                hits_at_100_head += 1
            if ranks_head[i] <= 10:
                hits_at_10_head += 1
            if ranks_head[i] <= 3:
                hits_at_3_head += 1
            if ranks_head[i] == 1:
                hits_at_1_head += 1

        for i in range(len(ranks_tail)):
            if ranks_tail[i] <= 100:
                hits_at_100_tail += 1
            if ranks_tail[i] <= 10:
                hits_at_10_tail += 1
            if ranks_tail[i] <= 3:
                hits_at_3_tail += 1
            if ranks_tail[i] == 1:
                hits_at_1_tail += 1

        assert len(ranks_head) == len(reciprocal_ranks_head)
        assert len(ranks_tail) == len(reciprocal_ranks_tail)

        hits_100_head = hits_at_100_head / len(ranks_head)
        hits_10_head = hits_at_10_head / len(ranks_head)
        hits_3_head = hits_at_3_head / len(ranks_head)
        hits_1_head = hits_at_1_head / len(ranks_head)
        mean_rank_head = sum(ranks_head) / len(ranks_head)
        mean_reciprocal_rank_head = sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)

        hits_100_tail = hits_at_100_tail / len(ranks_tail)
        hits_10_tail = hits_at_10_tail / len(ranks_tail)
        hits_3_tail = hits_at_3_tail / len(ranks_tail)
        hits_1_tail = hits_at_1_tail / len(ranks_tail)
        mean_rank_tail = sum(ranks_tail) / len(ranks_tail)
        mean_reciprocal_rank_tail = sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)

        hits_100 = (hits_at_100_head / len(ranks_head) + hits_at_100_tail / len(ranks_tail)) / 2
        hits_10 = (hits_at_10_head / len(ranks_head) + hits_at_10_tail / len(ranks_tail)) / 2
        hits_3 = (hits_at_3_head / len(ranks_head) + hits_at_3_tail / len(ranks_tail)) / 2
        hits_1 = (hits_at_1_head / len(ranks_head) + hits_at_1_tail / len(ranks_tail)) / 2
        mean_rank = (sum(ranks_head) / len(ranks_head) + sum(ranks_tail) / len(ranks_tail)) / 2
        mean_reciprocal_rank = (sum(reciprocal_ranks_head) / len(reciprocal_ranks_head) + sum(
            reciprocal_ranks_tail) / len(reciprocal_ranks_tail)) / 2

        metrics = {
            "Hits@100": hits_100,
            "Hits@10": hits_10,
            "Hits@3": hits_3,
            "Hits@1": hits_1,
            "MR": mean_rank,
            "MRR": mean_reciprocal_rank
        }

        return metrics, rel_pred_dict


class ConvKBCorpus(Corpus):
    def __init__(self, args, train_data, val_data, test_data, entity2id, relation2id):
        super(ConvKBCorpus, self).__init__(args, train_data, val_data, test_data, entity2id, relation2id)
        self.neg_num = args.neg_num
        if len(self.train_triples) % self.batch_size == 0:
            self.max_batch_num = len(self.train_triples) // self.batch_size
        else:
            self.max_batch_num = len(self.train_triples) // self.batch_size + 1

        self.train_indices = np.array(self.train_triples).astype(np.int32)
        self.train_values = np.array([[1]] * len(self.train_triples)).astype(np.float32)
        self.val_indices = np.array(self.val_triples).astype(np.int32)
        self.val_values = np.array([[1]] * len(self.val_triples)).astype(np.float32)
        self.test_indices = np.array(self.test_triples).astype(np.int32)
        self.test_values = np.array([[1]] * len(self.test_triples)).astype(np.float32)

        self.unique_entities = [entity2id[i] for i in train_data[2]]
        self.all_triples = {j: i for i, j in enumerate(self.train_triples + self.val_triples + self.test_triples)}

        self.batch_indices = np.empty((self.batch_size * (self.neg_num + 1), 3)).astype(np.int32)
        self.batch_values = np.empty((self.batch_size * (self.neg_num + 1), 1)).astype(np.float32)

    def shuffle(self):
        np.random.shuffle(self.train_indices)

    def get_batch(self, batch_num):
        if (batch_num + 1) * self.batch_size <= len(self.train_indices):
            self.batch_indices = np.empty((self.batch_size * (self.neg_num + 1), 3)).astype(np.int32)
            self.batch_values = np.empty((self.batch_size * (self.neg_num + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * batch_num, self.batch_size * (batch_num + 1))
            last_index = self.batch_size

        else:
            last_batch_size = len(self.train_indices) - self.batch_size * batch_num
            self.batch_indices = np.empty((last_batch_size * (self.neg_num + 1), 3)).astype(np.int32)
            self.batch_values = np.empty((last_batch_size * (self.neg_num + 1), 1)).astype(np.float32)

            indices = range(self.batch_size * batch_num, len(self.train_indices))
            last_index = last_batch_size

        self.batch_indices[:last_index, :] = self.train_indices[indices, :]
        self.batch_values[:last_index, :] = self.train_values[indices, :]
        random_entities = np.random.randint(0, len(self.entity2id), last_index * self.neg_num)
        self.batch_indices[last_index: (last_index * (self.neg_num + 1)), :] = np.tile(
            self.batch_indices[:last_index, :], (self.neg_num, 1))
        self.batch_values[last_index: (last_index * (self.neg_num + 1)), :] = np.tile(
            self.batch_values[:last_index, :], (self.neg_num, 1))

        for i in range(last_index):
            for j in range(self.neg_num // 2):
                current_index = i * (self.neg_num // 2) + j

                while(random_entities[current_index], self.batch_indices[last_index + current_index, 1],
                      self.batch_indices[last_index + current_index, 2]) in self.all_triples.keys():
                    random_entities[current_index] = np.random.randint(0, len(self.entity2id))

                self.batch_indices[last_index + current_index, 0] = random_entities[current_index]
                self.batch_values[last_index + current_index, :] = [-1]
            for j in range(self.neg_num // 2):
                current_index = last_index * (self.neg_num // 2) + i * (self.neg_num // 2) + j

                while (self.batch_indices[last_index + current_index, 0], self.batch_indices[last_index + current_index, 1],
                       random_entities[current_index]) in self.all_triples.keys():
                    random_entities[current_index] = np.random.randint(0, len(self.entity2id))

                self.batch_indices[last_index + current_index, 2] = random_entities[current_index]
                self.batch_values[last_index + current_index, :] = [-1]

        return self.batch_indices, self.batch_values

    def get_validation_pred(self, model, split='test'):
        ranks_head, ranks_tail = [], []
        reciprocal_ranks_head, reciprocal_ranks_tail = [], []
        hits_at_100_head, hits_at_100_tail = 0, 0
        hits_at_10_head, hits_at_10_tail = 0, 0
        hits_at_3_head, hits_at_3_tail = 0, 0
        hits_at_1_head, hits_at_1_tail = 0, 0
        entity_list = [i for i in self.entity2id.values()]
        if split == 'val':
            split_triples = np.array(self.val_triples).astype(np.int32)
        elif split == 'test':
            split_triples = np.array(self.test_triples).astype(np.int32)

        for i in range(split_triples.shape[0]):
            if split_triples[i, 0] not in self.unique_entities or split_triples[i, 2] not in self.unique_entities:
                continue
            x_head = np.tile(split_triples[i, :], (len(self.entity2id), 1))
            x_tail = np.tile(split_triples[i, :], (len(self.entity2id), 1))
            x_head[:, 0] = entity_list
            x_tail[:, 2] = entity_list

            last_index_head, last_index_tail = [], []
            for idx in range(len(x_head)):
                head = (x_head[idx][0], x_head[idx][1], x_head[idx][2])
                if head in self.all_triples.keys():
                    last_index_head.append(idx)

                tail = (x_tail[idx][0], x_tail[idx][1], x_tail[idx][2])
                if tail in self.all_triples.keys():
                    last_index_tail.append(idx)

            x_head = np.delete(x_head, last_index_head, axis=0)
            x_tail = np.delete(x_tail, last_index_tail, axis=0)
            rand_head = np.random.randint(x_head.shape[0])
            rand_tail = np.random.randint(x_tail.shape[0])
            x_head = np.insert(x_head, rand_head, split_triples[i], axis=0)
            x_tail = np.insert(x_tail, rand_tail, split_triples[i], axis=0)
            x_head = torch.LongTensor(x_head).to(self.device)
            x_tail = torch.LongTensor(x_tail).to(self.device)
            #scores_head = model.forward(x_head)
            scores_head = model.predict(x_head)
            sorted_scores_head, sorted_triples_head = torch.sort(scores_head.view(-1), dim=-1, descending=True)
            ranks_head.append(np.where(sorted_triples_head.cpu().numpy() == rand_head)[0][0]+1)
            reciprocal_ranks_head.append(1.0 / ranks_head[-1])
            #scores_tail = model.forward(x_tail)
            scores_tail = model.predict(x_tail)
            sorted_scores_tail, sorted_triples_tail = torch.sort(scores_tail.view(-1), dim=-1, descending=True)
            ranks_tail.append(np.where(sorted_triples_tail.cpu().numpy() == rand_tail)[0][0]+1)
            reciprocal_ranks_tail.append(1.0 / ranks_tail[-1])

        for i in range(len(ranks_head)):
            if ranks_head[i] <= 100:
                hits_at_100_head += 1
            if ranks_head[i] <= 10:
                hits_at_10_head += 1
            if ranks_head[i] <= 3:
                hits_at_3_head += 1
            if ranks_head[i] == 1:
                hits_at_1_head += 1

        for i in range(len(ranks_tail)):
            if ranks_tail[i] <= 100:
                hits_at_100_tail += 1
            if ranks_tail[i] <= 10:
                hits_at_10_tail += 1
            if ranks_tail[i] <= 3:
                hits_at_3_tail += 1
            if ranks_tail[i] == 1:
                hits_at_1_tail += 1

        assert len(ranks_head) == len(reciprocal_ranks_head)
        assert len(ranks_tail) == len(reciprocal_ranks_tail)

        hits_100_head = hits_at_100_head / len(ranks_head)
        hits_10_head = hits_at_10_head / len(ranks_head)
        hits_3_head = hits_at_3_head / len(ranks_head)
        hits_1_head = hits_at_1_head / len(ranks_head)
        mean_rank_head = sum(ranks_head) / len(ranks_head)
        mean_reciprocal_rank_head = sum(reciprocal_ranks_head) / len(reciprocal_ranks_head)

        hits_100_tail = hits_at_100_tail / len(ranks_tail)
        hits_10_tail = hits_at_10_tail / len(ranks_tail)
        hits_3_tail = hits_at_3_tail / len(ranks_tail)
        hits_1_tail = hits_at_1_tail / len(ranks_tail)
        mean_rank_tail = sum(ranks_tail) / len(ranks_tail)
        mean_reciprocal_rank_tail = sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)

        hits_100 = (hits_at_100_head / len(ranks_head) + hits_at_100_tail / len(ranks_tail)) / 2
        hits_10 = (hits_at_10_head / len(ranks_head) + hits_at_10_tail / len(ranks_tail)) / 2
        hits_3 = (hits_at_3_head / len(ranks_head) + hits_at_3_tail / len(ranks_tail)) / 2
        hits_1 = (hits_at_1_head / len(ranks_head) + hits_at_1_tail / len(ranks_tail)) / 2
        mean_rank = (sum(ranks_head) / len(ranks_head) + sum(ranks_tail) / len(ranks_tail)) / 2
        mean_reciprocal_rank = (sum(reciprocal_ranks_head) / len(reciprocal_ranks_head) + sum(reciprocal_ranks_tail) / len(reciprocal_ranks_tail)) / 2

        metrics = {
            "Hits@100_head": hits_100_head,
            "Hits@10_head": hits_10_head,
            "Hits@3_head": hits_3_head,
            "Hits@1_head": hits_1_head,
            "Mean Rank_head": mean_rank_head,
            "Mean Reciprocal Rank_head": mean_reciprocal_rank_head,
            "Hits@100_tail": hits_100_tail, "Hits@10_tail": hits_10_tail, "Hits@3_tail": hits_3_tail, "Hits@1_tail": hits_1_tail,
            "Mean Rank_tail": mean_rank_tail, "Mean Reciprocal Rank_tail": mean_reciprocal_rank_tail,
            "Hits@100": hits_100, "Hits@10": hits_10, "Hits@3": hits_3, "Hits@1": hits_1,
            "Mean Rank": mean_rank, "Mean Reciprocal Rank": mean_reciprocal_rank}

        return metrics


