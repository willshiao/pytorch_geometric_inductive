from collections import namedtuple
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.data.storage import BaseStorage, NodeStorage
from torch_geometric.transforms import BaseTransform
from torch_geometric.transforms.random_link_split import RandomLinkSplit
from torch_geometric.transforms.random_node_split import RandomNodeSplit
from torch_geometric.utils import negative_sampling, subgraph


def _create_mask(base_mask, rows, cols):
    return base_mask[rows] & base_mask[cols]


def _split_edges(edge_index, val_ratio, test_ratio):
    mask = edge_index[0] <= edge_index[1]
    perm = mask.nonzero(as_tuple=False).view(-1)
    perm = perm[torch.randperm(perm.size(0), device=perm.device)]
    num_val = int(val_ratio * perm.numel())
    num_test = int(test_ratio * perm.numel())

    num_train = perm.numel() - num_val - num_test
    train_edges = perm[:num_train]
    val_edges = perm[num_train:num_train + num_val]
    test_edges = perm[num_train + num_val:]
    train_edge_index = edge_index[:, train_edges]
    train_edge_index = torch.cat([train_edge_index, train_edge_index.flip([0])], dim=-1)
    val_edge_index = edge_index[:, val_edges]
    val_edge_index = torch.cat([val_edge_index, val_edge_index.flip([0])], dim=-1)
    test_edge_index = edge_index[:, test_edges]

    return train_edge_index, val_edge_index, test_edge_index


InductiveTestEdges = namedtuple('InductiveTestEdges',
                                ['old_old', 'old_new', 'new_new', 'all', 'negative_samples'])
InductiveTestEdges.__doc__ = '''\
A :class:`~collections.namedtuple` of the different edges types.
Note that the fields can also be accessed using their numeric indices since this class is a namedtuple.

Contains:
    :obj:`old_old` (:class:`~torch.Tensor`): Edge index of only old-old links.
    
    :obj:`old_new` (:class:`~torch.Tensor`): Edge index of only old-new links.
    
    :obj:`new_new` (:class:`~torch.Tensor`): Edge index of only new-new links.
    
    :obj:`all` (:class:`~torch.Tensor`): Edge index of all of the previous three.
    
    :obj:`negative_samples` (:class:`~torch.Tensor`): Edge index of negative
        samples containing a mixture of old-old, old-new, and new-new edge types.
        This is provided for testing purposes.'''


@functional_transform('random_inductive_split')
class RandomInductiveSplit(BaseTransform):
    r"""Performs the inductive split used in
    `"Link Prediction with Non-Contrastive Learning"
    <https://openreview.net/forum?id=9Jaz4APHtWD>`_ and
    `"Linkless Link Prediction via Relational Distillation"
    <https://arxiv.org/abs/2210.05801>`_.
    This transform partitions the data into four groups:
    training, validation, inference, and test datasets.
    The training data is used for training the model and the inference data
    is used purely for message passing before being evaluated on the test
    dataset. The inference data also contains new nodes.

    This method also returns a :class:`~torch_geometric.transforms.random_inductive_split.InductiveTestEdges`
    object, which contains the test edges grouped into "old-old", "old-new",
    and "new-new" edges. We refer to nodes present in the training set as
    "old" nodes and node present only in the inference and testing sets as
    "new" nodes. Please see the linked papers above for a more detailed
    description of this setting.

    Args:
        test_ratio (float, optional): The ratio of edges to be used as training
            examples.
            (default: :obj:`0.1`)
        new_node_ratio (float, optional): The ratio of nodes to be taken as "new"
            nodes. These nodes will only be present in the inference and testing
            datasets and will not be in the training dataset. Note that all of the
            edges to/from a new node will be used for inference or testing.
            (default: :obj:`0.1`)
        val_ratio (float, optional): The ratio of validation edges taken from the
            training data. Can be set to :obj:`0` if no validation edges are required.
            (default: :obj:`0.1`)
        old_old_ratio (float, optional):
            TODO(willshiao): what is the purpose of my existence?
            (default: :obj:`0.1`)
    """

    def __init__(
            self,
            test_ratio: float = 0.1,
            new_node_ratio: float = 0.1,
            val_ratio: float = 0.1,
            # TODO(willshiao): revisit name
            old_old_ratio: float = 0.1):
        assert (0 < test_ratio <= 1)
        assert (0 < new_node_ratio <= 1)
        assert (0 < val_ratio <= 1)
        assert (0 < old_old_ratio <= 1)
        self.old_old_ratio = old_old_ratio
        self.val_ratio = val_ratio
        self.new_node_ratio = new_node_ratio
        self.test_ratio = test_ratio

    def __call__(
        self,
        data: Data,
    ) -> Tuple[Data, Data, Data, InductiveTestEdges]:
        return self._split(data)

    def _split(self, data: Data) -> Tuple[Data, Data, Data, InductiveTestEdges]:
        # sample some negatives to use globally
        num_negatives = round(self.test_ratio * data.edge_index.size(1) / 2)
        negative_samples = negative_sampling(data.edge_index,
                                             data.num_nodes,
                                             num_negatives,
                                             force_undirected=True)

        # Split the nodes into "old" and "new" nodes
        node_splitter = RandomNodeSplit(num_val=0.0, num_test=self.new_node_ratio)
        new_data = node_splitter(data)

        # Separate the edges between old nodes
        rows, cols = new_data.edge_index
        old_old_edges = _create_mask(new_data.train_mask, rows, cols)
        old_old_ei = new_data.edge_index[:, old_old_edges]
        old_old_train, old_old_val, old_old_test = _split_edges(old_old_ei, self.old_old_ratio,
                                                                self.test_ratio)

        # Separate the edges between old and new nodes
        old_new_edges = (new_data.train_mask[rows] & new_data.test_mask[cols]) | (
            new_data.test_mask[rows] & new_data.train_mask[cols])
        old_new_ei = new_data.edge_index[:, old_new_edges]
        old_new_train, _, old_new_test = _split_edges(old_new_ei, 0.0, self.test_ratio)

        # Separate the edges between new and new nodes
        new_new_edges = _create_mask(new_data.test_mask, rows, cols)
        new_new_ei = new_data.edge_index[:, new_new_edges]
        new_new_train, _, new_new_test = _split_edges(new_new_ei, 0.0, self.test_ratio)

        # Create a bundle of all of the different testing sets
        # (old-old, old-new, new-new, all testing, negative_samples)
        test_edge_index = torch.cat([old_old_test, old_new_test, new_new_test], dim=-1)
        test_edge_bundle = InductiveTestEdges(old_old_test, old_new_test, new_new_test,
                                              test_edge_index, negative_samples)

        # Use the induced subgraph of only the old-old nodes
        training_only_ei = subgraph(new_data.train_mask, old_old_train, relabel_nodes=True)[0]
        training_only_x = new_data.x[new_data.train_mask]

        given_data = Data(training_only_x, training_only_ei)
        # Split the training data into train/val sets
        val_splitter = RandomLinkSplit(0.0, self.val_ratio, is_undirected=True)
        training_data, _, val_data = val_splitter(given_data)

        # Create the inference-only data.
        inference_edge_index = torch.cat([old_old_train, old_old_val, old_new_train, new_new_train],
                                         dim=-1)
        inference_data = Data(new_data.x, inference_edge_index)

        return training_data, val_data, inference_data, test_edge_bundle

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
