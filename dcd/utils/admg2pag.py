import os
import shutil
import pickle
import itertools
import numpy as np
from ananke.graphs import ADMG


# danish mod: added below function to return matrix representation of PAG
def get_pag_matrix(G):
    """
    Function to return a numpy ndarray representing the PAG

    :param G: Ananke ADMG with 'pag_edges' attribute.
    :return: numpy nd-array of shape (num_nodes, num_nodes) where columns are sources
             and rows are targets and A[i, j] represents the edge mark at node j
             1 = circle (o), 2 = head (>), 3 = tail (-)
    """
    num_nodes = len(G.vertices)
    pag_matrix = np.zeros((num_nodes, num_nodes))
    vertex_map = {v: i for i, v in enumerate(G.vertices)}  # we don't care which vertex is mapped to which number
    edge_mark_map = {"o": 1, "-": 3, ">": 2, "<": 2}
    for edge in G.pag_edges:
        pag_matrix[vertex_map[edge["v"]], vertex_map[edge["u"]]] = edge_mark_map[edge["type"][0]]
        pag_matrix[vertex_map[edge["u"]], vertex_map[edge["v"]]] = edge_mark_map[edge["type"][-1]]
    return pag_matrix

# danish mod: added below function to convert adjacency matrices to ananke-ADMG graph
def get_graph_from_adj(D, B):
    d = D.shape[0]
    vertices = [f'{i}' for i in range(d)]
    di_edges = [(f'{idx[1]}', f'{idx[0]}') for idx, x in np.ndenumerate(D) if x > 0]
    bi_edges = [(f'{idx[0]}', f'{idx[1]}') for idx, x in np.ndenumerate(np.triu(B, 1)) if x > 0]
    return ADMG(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges)

def pprint_pag(G):
    """
    Function to pretty print out PAG edges

    :param G: Ananke ADMG with 'pag_edges' attribute.
    :return: None.
    """
    print ('-'*10)
    print (f'Nodes: {list(G.vertices.keys())}')
    for edge in G.pag_edges:
        print (f'{edge["u"]} {edge["type"]} {edge["v"]}')


def write_admg_to_file(G, filename):
    """
    Function to write ADMG to file in correct format for PAG conversion.

    :param G: Ananke ADMG.
    :return: None.
    """

    # danish mod: added with close instead of open and close
    with open(filename, 'w') as f:
        f.write("Graph Nodes:\n")
        f.write(','.join(['X'+str(v) for v in G.vertices]) + '\n\n')
        f.write("Graph Edges:\n")
        counter = 1
        for Vi, Vj in G.di_edges:
            f.write(str(counter) + '. X' + str(Vi) + ' --> X' + str(Vj) + '\n')
            counter += 1
        for Vi, Vj in G.bi_edges:
            f.write(str(counter) + '. X' + str(Vi) + ' <-> X' + str(Vj) + '\n')
            counter += 1


def inducing_path(G, Vi, Vj):
    """
    Checks if there is an inducing path between Vi and Vj in G.

    :return: boolean indicator whether there is an inducing path.
    """

    # easy special case of directed adjacency
    if Vi in G.parents([Vj]) or Vj in G.parents([Vi]) or Vi in G.siblings([Vj]):
        return True

    ancestors_ViVj = G.ancestors([Vi, Vj])
    visit_stack = [s for s in G.siblings([Vi]) if s in ancestors_ViVj]
    visit_stack += [c for c in G.children([Vi]) if c in ancestors_ViVj]
    visited = set()

    while visit_stack:

        if Vj in visit_stack or Vj in G.parents(visit_stack):
            return True

        v = visit_stack.pop()
        visited.add(v)
        visit_stack.extend(set([s for s in G.siblings([v]) if s in ancestors_ViVj]) - visited)
    return False


def mag_projection(G):
    """
    Get MAG projection of an ADMG G.

    :param G: Ananke ADMG.
    :return: Ananke ADMG corresponding to a MAG.
    """

    G_mag = ADMG(G.vertices)

    # iterate over all vertex pairs
    for Vi, Vj in itertools.combinations(G.vertices, 2):

        # check if there is an inducing path
        if inducing_path(G, Vi, Vj):
            # connect based on ancestrality
            if Vi in G.ancestors([Vj]):
                G_mag.add_diedge(Vi, Vj)
            elif Vj in G.ancestors([Vi]):
                G_mag.add_diedge(Vj, Vi)
            else:
                G_mag.add_biedge(Vi, Vj)

    return G_mag


# danish mod: changed from tmp to dcd/tmp
def admg_to_pag(G, tmpdir='dcd/tmp'):
    """
    Write an ADMG G to file, and then convert it to a PAG using tetrad

    :param G: Ananke ADMG.
    :return: Ananke ADMG with an 'pag_edges' attribute corresponding to a PAG.
    """

    os.makedirs(f'{tmpdir}/', exist_ok=True)

    # write to disk for tetrad
    mag = mag_projection(G)
    # danish mod: drawing the MAG
    # mag.draw(direction="LR").render(filename='0399_mag_graph', directory=folder, format='pdf')
    write_admg_to_file(mag, f'{tmpdir}/G.mag')

    # convert to pag and write to disk
    # danish mod: updated to latest version number
    # danish mod: added dcd to path
    os.system(f'java -classpath "dcd/utils/tetrad-lib-7.6.9.jar:dcd/utils/xom-1.3.5.jar:dcd/utils/commons-lang3-3.20.0.jar:dcd/utils/" convertMag2Pag {tmpdir}/G.mag')

    # load back into new ADMG and return
    # danish mod: added with clause for file open / close
    with open(f'{tmpdir}/G.mag.pag', 'r') as f:
        lines = f.read().strip().split('\n')
    # lines = open(f'{tmpdir}/G.mag.pag', 'r').read().strip().split('\n')
    nodes = lines[1].split(';')
    nodes = [str(node[1:]) for node in nodes] # remove X

    edges = []
    for line in lines[4:]:
      edge = line.split('. ')[1].split(' ')
      edges.append({'u':str(edge[0][1:]), 'v':str(edge[2][1:]), 'type':edge[1]})

    G = ADMG(nodes)
    G.pag_edges = edges

    # cleanup
    # danish mod: commented out below line to prevent deletion, allowing debugging
    # shutil.rmtree(tmpdir)

    return G

# danish mod: added below code to be able to directly run the file if needed (for testing)
if __name__ == '__main__':
    import numpy as np
    import logging
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    # this is the bow-free ADMG from figure 1(c) in file 2021-differentiable-cd-c-unmeasured-confounding.pdf
    dim = 4
    folder = r'/mnt/windows/Users/lordh/Documents/LibraryOfBabel/Miscellany/Danish/linux_temp/temp_files/'
    D = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [1, 0, 0, 0],
                  [0, 0, 1, 0]])
    B = np.array([[0, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 0]])
    vertices = [f'{i}' for i in range(dim)]
    di_edges = [(f'{idx[1]}', f'{idx[0]}') for idx, x in np.ndenumerate(D) if x > 0]
    bi_edges = [(f'{idx[0]}', f'{idx[1]}') for idx, x in np.ndenumerate(np.triu(B, 1)) if x > 0]
    G = ADMG(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges)
    logging.info("made org graph")
    # G.draw(direction="LR").render(filename='0398_org_graph', directory=folder, format='pdf')
    # logging.info("drawn org graph")
    PAG = admg_to_pag(G)
    logging.info("converted org graph to pag")
    pprint_pag(PAG)
    print(f'PAG MATRIX:\n{get_pag_matrix(PAG)}')
    # PAG.draw(direction="LR").render(filename='0400_pag_graph', directory=folder, format='pdf')
    # logging.info("drawn pag graph")
