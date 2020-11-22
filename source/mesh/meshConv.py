import tensorflow as tf
import numpy as np

class meshConv(tf.Module):
    """ Computes convolution between edges and 4 incident (1-ring) edge neighbors
    in the forward pass takes:
    edge_feats: edge features (Batch x Features x Edges)
    mesh: list of mesh data-structure (len(mesh) == Batch)
    and applies convolution
    """
    def __init__(self, out_channels, k=5, bias=True):
        super(meshConv, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(1, k), bias_initializer=bias)
        self.k = k

    def call(self, edge_feats, mesh):
        """Perform forward pass on edge features"""
        edge_feats = tf.squeeze(edge_feats,-1)
        G = tf.concat([self.pad_neighbors(i, edge_feats.shape[2]) for i in mesh], axis=0)
        edge_feats = self.conv(G)
        return edge_feats

    def flatten_neighbors_indices(self, G):
        (b, ne, nn) = G.shape
        ne += 1
        batch_n = tf.math.floor(tf.range(b * ne).float() / ne).view(b, ne)
        add_fac = batch_n * ne
        add_fac = add_fac.view(b, ne, 1)
        add_fac = add_fac.repeat(1, 1, nn)
        G = G.float() + add_fac[:, 1:, :]
        return G

    
    def create_neighbors(self, edge_feats, G):
        """ gathers the edge features (x) with from the 1-ring indices (Gi)
        applys symmetric functions to handle order invariance
        returns a 'fake image' which can use 2d convolution on
        output dimensions: Batch x Channels x Edges x 5
        """
        Gshape = G.shape
        padding = tf.zeros((edge_feats.shape[0], edge_feats.shape[1], 1))
        edge_feats = tf.concat([padding, edge_feats], axis=2)
        G = G+1
        # first flatten indices
        G_flat = self.flatten_neighbors_indices(G)
        G_flat = G_flat.view(-1).long()
        #
        odim = edge_feats.shape
        edge_feats = edge_feats.permute(0, 2, 1).contiguous()
        edge_feats = edge_feats.view(odim[0] * odim[2], odim[1])
        
        f = tf.gather_nd(edge_feats, G_flat)                                                                                                                                                                                                                                                                                                                  n(x, dim=0, index=Gi_flat)
        f = f.view(Gshape[0], Gshape[1], Gshape[2], -1)
        f = f.permute(0, 3, 1, 2)

        # apply the symmetric functions for an equivariant conv
        x_1 = f[:, :, :, 1] + f[:, :, :, 3]
        x_2 = f[:, :, :, 2] + f[:, :, :, 4]
        x_3 = tf.math.abs(f[:, :, :, 1] - f[:, :, :, 3])
        x_4 = tf.math.abs(f[:, :, :, 2] - f[:, :, :, 4])
        f = tf.stack([f[:, :, :, 0], x_1, x_2, x_3, x_4], 3)
        
    
    def pad_neighbors(self, mesh, size):
        """ extracts one-ring neighbors (4x) -> mesh.edge_to_neighbors
        which is of size #edges x 4
        add the edge_id itself to make #edges x 5
        then pad to desired size e.g., size x 5
        """
        padded_neighbors = tf.tensor(mesh.edge_to_neighbors).float()
        padded_neighbors = tf.concat((tf.expand_dims(tf.range(len(mesh.edges)).float(), 1), padded_neighbors), axis=1)
        padded_neighbors = tf.pad(padded_neighbors, [0, 0, 0, size-len(mesh.edges)], 'CONSTANT')
        padded_neighbors = tf.expand_dims(padded_neighbors, 0)
        return padded_neighbors


    

