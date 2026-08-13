"""Train GNN to produce entity / relation / text / image embeddings incorporating KG structure.

Saves:
- datasets/<DATASET>/entity_embeddings_gnn.pth   (num_entities x embed_dim)
- datasets/<DATASET>/relation_embeddings_gnn.pth (num_relations x embed_dim)
- datasets/<DATASET>/text_embeddings_gnn.pth     (num_entities x text_dim)
- datasets/<DATASET>/img_embeddings_gnn.pth      (num_entities x img_dim)

Usage examples:
python scripts/train_gnn_embeddings.py --dataset MKG-W --epochs 5 --device cuda:0
"""
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class TriplesDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]


def load_mappings(data_dir):
    ent_path = Path(data_dir) / 'entity2id.txt'
    rel_path = Path(data_dir) / 'relation2id.txt'
    entity2id = {}
    relation2id = {}
    with open(ent_path) as f:
        for line in f:
            k, v = line.strip().split()
            entity2id[k] = int(v)
    with open(rel_path) as f:
        for line in f:
            k, v = line.strip().split()
            relation2id[k] = int(v)
    return entity2id, relation2id


def load_triples(data_dir, entity2id, relation2id, file='train.txt', add_reverse=True, num_rel_types=None):
    triples = []
    with open(Path(data_dir) / file) as f:
        for line in f:
            if not line.strip():
                continue
            s, r, o, _ = line.strip().split()
            rid = relation2id[r]
            triples.append((entity2id[s], rid, entity2id[o]))
            # add reverse triple as a distinct relation type
            if add_reverse:
                if num_rel_types is None:
                    raise ValueError('num_rel_types must be provided when add_reverse is True')
                rev_rid = rid + num_rel_types
                triples.append((entity2id[o], rev_rid, entity2id[s]))
    return triples


def build_adj(num_nodes, triples):
    # Directed edges from triples (s -> o). Each triple represents a directed connection.
    row = []
    col = []
    for s, r, o in triples:
        row.append(s)
        col.append(o)
    indices = torch.tensor([row, col], dtype=torch.long)
    vals = torch.ones(indices.size(1), dtype=torch.float)
    adj = torch.sparse_coo_tensor(indices, vals, (num_nodes, num_nodes))
    # add self loops
    idx = torch.arange(0, num_nodes, dtype=torch.long)
    self_idx = torch.stack([idx, idx], dim=0)
    adj = adj.coalesce()
    indices = torch.cat([adj.indices(), self_idx], dim=1)
    vals = torch.cat([adj.values(), torch.ones(num_nodes)], dim=0)
    adj = torch.sparse_coo_tensor(indices, vals, (num_nodes, num_nodes)).coalesce()
    # normalize: D^-1/2 A D^-1/2
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    i = adj.indices()
    v = adj.values()
    v = v * deg_inv_sqrt[i[0]] * deg_inv_sqrt[i[1]]
    adj_norm = torch.sparse_coo_tensor(i, v, adj.size())
    return adj_norm.coalesce()


class SimpleGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_rel, rel_dim, text_dim, img_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        # relation embedding size may differ from entity out_dim; keep relation embeddings in `rel_dim`
        self.rel_emb = nn.Embedding(num_rel, rel_dim)
        # map relation embedding to entity space for scoring when dims differ
        if rel_dim != out_dim:
            self.rel_map = nn.Linear(rel_dim, out_dim)
        else:
            self.rel_map = None
        # projectors to original feature spaces
        self.to_text = nn.Linear(out_dim, text_dim)
        self.to_img = nn.Linear(out_dim, img_dim)

    def forward(self, x, adj):
        # x: N x in_dim
        h = F.relu(self.lin1(x))
        h = self.propagate(h, adj)
        h = self.lin2(h)
        h = self.propagate(h, adj)
        return h

    @staticmethod
    def propagate(x, adj):
        # sparse-dense matmul
        return torch.sparse.mm(adj, x)


def score_triples(ent_emb, rel_emb, rel_map, s_idx, r_idx, o_idx):
    # DistMult-like: score = <e_s * r_mapped, e_o>
    e_s = ent_emb[s_idx]
    e_o = ent_emb[o_idx]
    r = rel_emb[r_idx]
    if rel_map is not None:
        r = rel_map(r)
    return torch.sum(e_s * r * e_o, dim=-1)


def train(args):
    data_dir = Path(args.data_dir) / args.dataset
    entity2id, relation2id = load_mappings(data_dir)
    num_entities = len(entity2id)
    num_rel_types = len(relation2id)
    # in directed setting each original relation has a reverse relation id = id + num_rel_types
    num_rel = num_rel_types * 2

    # load pretrained text/img features
    text_feat = torch.load(data_dir / 'text_features.pth') if (data_dir / 'text_features.pth').exists() else None
    img_feat = torch.load(data_dir / 'img_features.pth') if (data_dir / 'img_features.pth').exists() else None

    if text_feat is None and img_feat is None:
        # fallback to learnable embeddings
        in_dim = args.embed_dim
        x = torch.randn(num_entities, in_dim)
    else:
        # align sizes
        if text_feat is None:
            text_dim = 0
        else:
            text_dim = text_feat.size(1)
        if img_feat is None:
            img_dim = 0
        else:
            img_dim = img_feat.size(1)
        if text_feat is None:
            x = img_feat
        elif img_feat is None:
            x = text_feat
        else:
            x = torch.cat([text_feat, img_feat], dim=1)
        in_dim = x.size(1)

    triples = load_triples(data_dir, entity2id, relation2id, file='train.txt', add_reverse=True, num_rel_types=num_rel_types)
    print(f"Loaded {len(triples)} triples, {num_entities} entities, {num_rel} relations")
    adj = build_adj(num_entities, triples)

    device = torch.device(args.device)
    x = x.to(device)
    adj = adj.to(device)
    if text_feat is not None:
        text_feat = text_feat.to(device)
    if img_feat is not None:
        img_feat = img_feat.to(device)

    model = SimpleGCN(in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=args.embed_dim, num_rel=num_rel,
                      rel_dim=args.rel_dim,
                      text_dim=(text_feat.size(1) if text_feat is not None else 0),
                      img_dim=(img_feat.size(1) if img_feat is not None else 0))
    model = model.to(device)
    rel_emb = model.rel_emb

    # separate GNNs for text and image to preserve modality-specific features
    text_model = None
    img_model = None
    text_to_ent = None
    img_to_ent = None
    if text_feat is not None:
        text_model = SimpleGCN(in_dim=text_feat.size(1), hidden_dim=args.hidden_dim, out_dim=text_feat.size(1), num_rel=num_rel,
                               rel_dim=args.rel_dim, text_dim=text_feat.size(1), img_dim=0)
        text_model = text_model.to(device)
        text_to_ent = nn.Linear(text_feat.size(1), args.embed_dim).to(device)
    if img_feat is not None:
        img_model = SimpleGCN(in_dim=img_feat.size(1), hidden_dim=args.hidden_dim, out_dim=img_feat.size(1), num_rel=num_rel,
                              rel_dim=args.rel_dim, text_dim=0, img_dim=img_feat.size(1))
        img_model = img_model.to(device)
        img_to_ent = nn.Linear(img_feat.size(1), args.embed_dim).to(device)

    # aggregate parameters for optimizer
    params = list(model.parameters())
    if text_model is not None:
        params += list(text_model.parameters()) + list(text_to_ent.parameters())
    if img_model is not None:
        params += list(img_model.parameters()) + list(img_to_ent.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    dataset = TriplesDataset(triples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    all_entity_idx = torch.arange(num_entities, device=device)

    for epoch in tqdm(range(args.epochs), desc='Epochs'):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f'Epoch {epoch+1}', leave=False):
            # convert batch (list of tuples) directly to tensor
            # print(f"batch: {batch}")
            # handle collated batch being a tuple of tensors (s_tensor, r_tensor, o_tensor)
            if isinstance(batch, (list, tuple)) and len(batch) == 3 and isinstance(batch[0], torch.Tensor):
                batch = torch.stack(batch, dim=1).to(device)
            else:
                try:
                    if isinstance(batch[0], (list, tuple)):
                        batch = torch.LongTensor(batch).to(device)
                    else:
                        # single element batch
                        batch = torch.LongTensor([batch]).to(device)
                except Exception:
                    # fallback: ensure all values are ints and build list
                    batch_list = []
                    for b in batch:
                        batch_list.append([int(x) for x in b])
                    batch = torch.LongTensor(batch_list).to(device)
            s = batch[:, 0]
            r = batch[:, 1]
            o = batch[:, 2]

            # forward to compute fresh embeddings
            ent_h = model(x, adj)  # N x D

            # positive scores
            pos_scores = score_triples(ent_h, rel_emb.weight, model.rel_map, s, r, o)

            # negative sampling: corrupt object
            neg_o = torch.randint(0, num_entities, (s.size(0), args.negatives), device=device)
            # expand s and r to match negatives
            s_exp = s.unsqueeze(1).expand(-1, args.negatives).reshape(-1)
            r_exp = r.unsqueeze(1).expand(-1, args.negatives).reshape(-1)
            neg_o_exp = neg_o.reshape(-1)
            neg_scores = score_triples(ent_h, rel_emb.weight, model.rel_map, s_exp, r_exp, neg_o_exp)
            neg_scores = neg_scores.view(s.size(0), args.negatives)

            # margin ranking loss: want pos > neg
            loss_pos = F.relu(args.margin - pos_scores.unsqueeze(1) + neg_scores).mean()

            # modality alignment losses
            loss_mod = 0.0
            if text_model is not None and args.alpha_text > 0:
                text_h = text_model(text_feat, adj)  # N x text_dim
                text_ent = text_to_ent(text_h)  # N x embed_dim
                l_text = F.mse_loss(text_ent, ent_h)
                loss_mod = loss_mod + args.alpha_text * l_text
            if img_model is not None and args.alpha_img > 0:
                img_h = img_model(img_feat, adj)  # N x img_dim
                img_ent = img_to_ent(img_h)  # N x embed_dim
                l_img = F.mse_loss(img_ent, ent_h)
                loss_mod = loss_mod + args.alpha_img * l_img

            loss = loss_pos + loss_mod

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * s.size(0)

        avg_loss = total_loss / len(dataset)
        tqdm.write(f"Epoch {epoch+1}/{args.epochs} avg_loss={avg_loss:.6f}")

        # Save best model (lower avg_loss is better). Overwrite existing files if present.
        if epoch == 0:
            best_loss = avg_loss
        if avg_loss <= best_loss:
            best_loss = avg_loss
            out_dir = data_dir
            ckpt_path = out_dir / 'model_gnn_best.pth'
            torch.save(model.state_dict(), ckpt_path)
            # also save current embeddings mapped to cpu
            with torch.no_grad():
                ent_h = model(x, adj)
                rel_raw = rel_emb.weight
                if getattr(model, 'rel_map', None) is not None:
                    rel_h = model.rel_map(rel_raw.to(device))
                else:
                    rel_h = rel_raw.to(device)
                # modality outputs: prefer dedicated modality models when available
                if text_model is not None:
                    text_h = text_model(text_feat, adj).cpu()
                else:
                    text_h = model.to_text(ent_h).cpu()
                if img_model is not None:
                    img_h = img_model(img_feat, adj).cpu()
                else:
                    img_h = model.to_img(ent_h).cpu()

            torch.save(ent_h.cpu(), out_dir / 'entity_embeddings_gnn.pth')
            torch.save(rel_h.cpu(), out_dir / 'relation_embeddings_gnn.pth')
            torch.save(text_h, out_dir / 'text_embeddings_gnn.pth')
            torch.save(img_h, out_dir / 'img_embeddings_gnn.pth')
            # print saved tensor shapes
            tqdm.write(f"Saved best model to {ckpt_path} (loss={best_loss:.6f})")
            tqdm.write(f"entity_embeddings_gnn.pth shape: {tuple(ent_h.cpu().shape)}")
            tqdm.write(f"relation_embeddings_gnn.pth shape: {tuple(rel_h.cpu().shape)}")
            tqdm.write(f"text_embeddings_gnn.pth shape: {tuple(text_h.shape)}")
            tqdm.write(f"img_embeddings_gnn.pth shape: {tuple(img_h.shape)}")

    # Save embeddings
    model.eval()
    with torch.no_grad():
        ent_h = model(x, adj)  # N x D on device
        rel_raw = rel_emb.weight
        if getattr(model, 'rel_map', None) is not None:
            rel_h = model.rel_map(rel_raw.to(device))
        else:
            rel_h = rel_raw.to(device)
        if text_model is not None:
            text_h = text_model(text_feat, adj).cpu()
        else:
            text_h = model.to_text(ent_h).cpu()
        if img_model is not None:
            img_h = img_model(img_feat, adj).cpu()
        else:
            img_h = model.to_img(ent_h).cpu()

    out_dir = data_dir
    torch.save(ent_h.cpu(), out_dir / 'entity_embeddings_gnn.pth')
    torch.save(rel_h.cpu(), out_dir / 'relation_embeddings_gnn.pth')
    torch.save(text_h, out_dir / 'text_embeddings_gnn.pth')
    torch.save(img_h, out_dir / 'img_embeddings_gnn.pth')

    print('Saved entity / relation / text / img embeddings to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='datasets')
    parser.add_argument('--dataset', type=str, default='MKG-W')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--negatives', type=int, default=5)
    parser.add_argument('--embed-dim', type=int, default=200, help='entity embedding output dim')
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--alpha-text', type=float, default=0.1, help='weight for text alignment loss')
    parser.add_argument('--alpha-img', type=float, default=0.1, help='weight for image alignment loss')
    parser.add_argument('--lr', type=float, default=1e-3)
    default_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default=default_dev)
    parser.add_argument('--margin', type=float, default=1.0)
    parser.add_argument('--rel-dim', type=int, default=256, help='dimension for relation embeddings')
    args = parser.parse_args()
    train(args)
