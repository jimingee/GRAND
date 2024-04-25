import os
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import Planetoid

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
DATA_PATH = f'{ROOT_DIR}/data'

def load_data(opt:dict, use_lcc:bool=True, dataset="Cora") -> InMemoryDataset:
    dataset = opt['dataset']
    print('Loading {} dataset...'.format(dataset))
    
    path = os.path.join(DATA_PATH, dataset)
    dataset = Planetoid(path, dataset)
    # origin dataset : Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
    
    if use_lcc:
        lcc = get_largest_connected_component(dataset)
        
        x_new = dataset.data.x[lcc]
        y_new = dataset.data.y[lcc]
        
        row, col = dataset.data.edge_index.numpy()
        edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
        edges = remap_edges(edges, get_node_mapper(lcc))
        
        data = Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool)
        )
        dataset.data = data
    # largest connected component : Data(x=[2485, 1433], edge_index=[2, 10138], y=[2485], train_mask=[2485], test_mask=[2485], val_mask=[2485])
        
    train_mask_exists = True
    
    if (use_lcc or not train_mask_exists) and not opt['geom_gcn_splits']:
        dataset.data = set_train_val_test_split(
            12345,
            dataset.data,
            num_development=1500)
        
    return dataset

        
def get_component(dataset:InMemoryDataset, start=0):
  # BFS 방식으로 start 노드와 연결된 connected 노드 얻음
  visited_nodes = set()
  queued_nodes = set([start])
  row, col = dataset.data.edge_index.numpy()
  while queued_nodes:
    current_node = queued_nodes.pop()
    visited_nodes.update([current_node])
    neighbors = col[np.where(row == current_node)[0]]
    neighbors = [n for n in neighbors if n not in visited_nodes and n not in queued_nodes]
    queued_nodes.update(neighbors)
  return visited_nodes


def get_largest_connected_component(dataset:InMemoryDataset):
  # 가장 큰 conneted_component의 node 번호 return 
  remaining_nodes = set(range(dataset.data.x.shape[0]))
  comps = []
  while remaining_nodes:
    start = min(remaining_nodes)
    comp = get_component(dataset, start)
    comps.append(comp)
    remaining_nodes = remaining_nodes.difference(comp)
  return np.array(list(comps[np.argmax(list(map(len, comps)))])) 


def get_node_mapper(lcc: np.ndarray) -> dict:
  mapper = {}
  counter = 0
  for node in lcc:
    mapper[node] = counter
    counter += 1
  return mapper


def remap_edges(edges: list, mapper: dict) -> list:
  row = [e[0] for e in edges]
  col = [e[1] for e in edges]
  row = list(map(lambda x: mapper[x], row))
  col = list(map(lambda x: mapper[x], col))
  return [row, col]


def set_train_val_test_split(seed: int, data: Data, num_development: int = 1500, num_per_class: int = 20) -> Data:
  rnd_state = np.random.RandomState(seed)
  num_nodes = data.y.shape[0]
  development_idx = rnd_state.choice(num_nodes, num_development, replace=False)
  test_idx = [i for i in np.arange(num_nodes) if i not in development_idx]
  
  train_idx = []
  rnd_state = np.random.RandomState(seed)
  for c in range(data.y.max() + 1): # y(클래스)의 수만큼 반복; c:1-7
    class_idx = development_idx[np.where(data.y[development_idx].cpu() == c)[0]]
    train_idx.extend(rnd_state.choice(class_idx, num_per_class, replace=False))

  val_idx = [i for i in development_idx if i not in train_idx]
  
  def get_mask(idx):
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask
  
  data.train_mask = get_mask(train_idx) 
  data.val_mask = get_mask(val_idx)   
  data.test_mask = get_mask(test_idx)  
  
  return data