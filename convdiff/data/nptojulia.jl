using PyCall
using JLD2
using GraphNeuralNetworks
using Delaunay

py"""
import pickle
 
def read_pickle(keys, path="./"):
    data_dict = {}
    for key in keys:
        with open(path+key+".pkl", "rb") as f:
            data_dict[key] = pickle.load(f)
    return data_dict
"""

py"""
import numpy as np
import torch
import torch_geometric

from torch_geometric.data import Data

from scipy.spatial import Delaunay

def generate_torchgeom_dataset(data, sig=0.0):
    n_sims = data['u'].shape[0]
    dataset = []

    for sim_ind in range(n_sims):
        print("{} / {}".format(sim_ind+1, n_sims))
        
        x = data['x'][sim_ind]
        tri = Delaunay(x)
        neighbors = neighbors_from_delaunay(tri)
        
        if sig > 0.0:
            print(f"Applying noise with sig={sig} to data")
            data['u'] += sig * np.random.randn(*data['u'].shape)
        
        # Find periodic couples and merge their neighborhoods
        origin_node = 0
        corner_nodes = []
        hor_couples = []
        vert_couples = []
        eps = 1.0e-6

        b = x.ravel().max()  # domain size

        for i in range(x.shape[0]):
            if is_near(x[i], [[b, 0], [0, b], [b, b]]):
                corner_nodes.append(i)
            elif is_near(x[i], [[0, 0]]):
                origin_node = i
            elif abs(x[i, 0]) < eps:  # left boundary
                for j in range(x.shape[0]):
                    if abs(x[j, 0] - b) < eps and abs(x[j, 1] - x[i, 1]) < eps:
                        hor_couples.append([i, j])
            elif abs(x[i, 1]) < eps:  # bottom boundary
                for j in range(x.shape[0]):
                    if abs(x[j, 1] - b) < eps and abs(x[j, 0] - x[i, 0]) < eps:
                        vert_couples.append([i, j])

        remove_nodes = []

        # Merge corners
        for i in corner_nodes:
            neighbors[origin_node].extend(neighbors[i])
            remove_nodes.append(i)

        # Merge horizontal couples
        for i, j in hor_couples:
            neighbors[i].extend(neighbors[j])
            remove_nodes.append(j)

        # Merge vertical couples
        for i, j in vert_couples:
            neighbors[i].extend(neighbors[j])
            remove_nodes.append(j)

        use_nodes = list(set(range(len(x))) - set(remove_nodes))

        # Remove right and top boundaries
        neighbors = np.array(neighbors, dtype=np.object)[use_nodes]

        # Rewrite indices of the removed nodes
        map_domain = corner_nodes + [x[1] for x in hor_couples] + [x[1] for x in vert_couples]
        map_codomain = [origin_node]*3 + [x[0] for x in hor_couples] + [x[0] for x in vert_couples]
        map_inds = dict(zip(map_domain, map_codomain))

        for i in range(len(neighbors)):
            for j in range(len(neighbors[i])):
                if neighbors[i][j] in remove_nodes:
                    neighbors[i][j] = map_inds[neighbors[i][j]]
            neighbors[i] = list(set(neighbors[i]))  # remove duplicates

        # Reset indices
        map_inds = dict(zip(use_nodes, range(len(use_nodes))))

        for i in range(len(neighbors)):
            for j in range(len(neighbors[i])):
                neighbors[i][j] = map_inds[neighbors[i][j]]

        # ...
        edge_index = []
        for i, _ in enumerate(neighbors):
            for _, neighbor in enumerate(neighbors[i]):
                if i == neighbor:
                    continue
                edge = [i, neighbor]
                edge_index.append(edge)
        edge_index = np.array(edge_index).T
        
        tg_data = {
            "x":data['u'][sim_ind, 0, use_nodes, :],
            "edge_index":edge_index,
            "y":data['u'][sim_ind][:, use_nodes],
            "pos":data['x'][sim_ind, use_nodes],
            "t":data['t'][sim_ind]
        }
        dataset.append(tg_data)

    return dataset



def is_near(x, y, eps=1.0e-16):
    x = np.array(x)
    y = np.array(y)
    for yi in y:
        if np.linalg.norm(x - yi) < eps:
            return True
    return False

def neighbors_from_delaunay(tri):
 
    neighbors_tri = tri.vertex_neighbor_vertices
    neighbors = []
    for i in range(len(neighbors_tri[0])-1):
        curr_node_neighbors = []
        for j in range(neighbors_tri[0][i], neighbors_tri[0][i+1]):
            curr_node_neighbors.append(neighbors_tri[1][j])
        neighbors.append(curr_node_neighbors)
    return neighbors
"""


read_pickle = py"read_pickle"
generate_torchgeom_dataset = py"generate_torchgeom_dataset"


train_data = read_pickle(["u","t","x"], path = "D:\\GitHub\\graphpdes_experiments\\convdiff\\data\\convdiff_2pi_n3000_t21_train\\")
train_data = generate_torchgeom_dataset(train_data, sig=0.0)

test_data = read_pickle(["u","t","x"], path = "D:\\GitHub\\graphpdes_experiments\\convdiff\\data\\convdiff_2pi_n3000_t21_test\\")
test_data = generate_torchgeom_dataset(test_data, sig=0.0)

function generata_dataset(data)
    num_samples = length(data)
    @show num_samples
    u = data[1]["y"]
    time_points = size(u,1)
    grid_size = size(u,2)
    u_train = Array{Float32}(undef, (grid_size,time_points, num_samples))
    u_train[:,:,1] = dropdims(u,dims = 3)'

    x = data[1]["pos"]
    s, t = data[1]["edge_index"][1,:] .+ 1, data[2]["edge_index"][2,:] .+ 1
    g = GNNGraph(s,t, ndata = (;x = Array{Float32}(x')))

    graphs = [g]
    for i in 2:num_samples
        u_train[:,:,i] = dropdims(u,dims = 3)'

        x_ =  Array{Float32}(data[i]["pos"]')
        s_, t_ = data[i]["edge_index"][1,:] .+ 1, data[i]["edge_index"][2,:] .+ 1
        g_ = GNNGraph(s_,t_, ndata = (;x = x_))
        push!(graphs,g_)
    end
    return graphs, u_train
end

gs_train, u_train = generata_dataset(train_data)
dt_train = Float32(train_data[1]["t"][2]-train_data[1]["t"][1])
tspan_train = (Float32(train_data[1]["t"][1]), Float32(train_data[1]["t"][end]))

gs_test, u_test = generata_dataset(test_data)
dt_test = Float32(test_data[1]["t"][2]-test_data[1]["t"][1])
tspan_test = (Float32(test_data[1]["t"][1]), Float32(test_data[1]["t"][end]))


jldsave("convdiff_n3000.jld2"; gs_train, dt_train, tspan_train, u_train, gs_test, dt_test, tspan_test, u_test)