from ...mesh.Mesh import Mesh
from ...mesh.EdgeConnection import EdgeConnection
from .CollapseSnapshot import CollapseSnapshot


def collapse_edge(mesh: Mesh, edge_key: int, snapshot: CollapseSnapshot):
    """Collapse an edge (two triangles) and update edge_mask to remove three
    edges: the one that's being collapsed and two edges that would overlap
    other edges as a result of the collapse. The mesh should be in a
    consistent state before and after this is called. Update the provided
    CollapseSnapshot with information about which edges have been collapsed.
    Note that the collapsed edge is parented to both surviving edges.

    Important: This edge collapse operation does not modify the faces. As a
    result, the mesh's faces will point to invalid vertices. This is because
    it's assumed that the collapse (pooling) operation will be accompanied by an
    unpooling operation, and that it won't be necessary to export a valid mesh
    created by this edge collapse. In other words, this edge collapse only
    updates the mesh with the information needed to pool edge-based features and
    determine which further edge collapses would be valid.

    Return True if the collapse succeeds. If the collapse would create
    non-manifold geometry, return False.
    """

    # Check for edge collapse validity.
    if not check_collapse_manifold(mesh, edge_key):
        return False

    # Collapse both sides (triangles).
    kept_vertex, masked_vertex = mesh.edges[edge_key]
    masked_edge_a = collapse_side(mesh, edge_key, 0, snapshot)
    masked_edge_b = collapse_side(mesh, edge_key, 2, snapshot)

    # Mask off the collapsed edge.
    mesh.edge_mask[edge_key] = False

    # Mask off the vertex that's being removed.
    mesh.vertex_mask[masked_vertex] = False

    # Move the kept vertex.
    mesh.vertices[kept_vertex, :] = (
        mesh.vertices[kept_vertex, :] + mesh.vertices[masked_vertex, :]
    ) * 0.5

    # This gets rid of references to the specified edge in vertex_to_edges.
    def discard_edge(discard_edge_key):
        vertex_a, vertex_b = mesh.edges[discard_edge_key]
        mesh.vertex_to_edges[vertex_a].remove(EdgeConnection(discard_edge_key, 0))
        mesh.vertex_to_edges[vertex_b].remove(EdgeConnection(discard_edge_key, 1))

    # Update the mesh's vertex_to_edges.
    masked_vertex_edges = mesh.vertex_to_edges[masked_vertex]
    discard_edge(edge_key)
    discard_edge(masked_edge_a)
    discard_edge(masked_edge_b)
    for edge_connection in masked_vertex_edges:
        mesh.edges[
            edge_connection.edge_index, edge_connection.index_in_edge
        ] = kept_vertex
    mesh.vertex_to_edges[kept_vertex].update(masked_vertex_edges)
    mesh.vertex_to_edges[masked_vertex] = None

    # Update the mesh's num_edges.
    mesh.num_edges -= 3

    return True


def check_collapse_manifold(mesh: Mesh, edge_key: int):
    """Return False if collapsing the specified edge would cause
    non-manifold geometry to be formed. There are two conditions that under
    which non-manifold geometry is created:
    1) A vertex that neighbors one of the edges has degree 3.
    2) The edge's vertices do not share exactly 2 vertex neighbors.
    """

    # If the edge doesn't exist anymore, it can't be collapsed.
    if mesh.edge_mask[edge_key] == False:
        return False

    vertex_a, vertex_b = mesh.edges[edge_key]

    # Get the one-ring neighbors around the edge's vertices.
    def get_neighbor_vertices(vertex):
        neighbor_has_degree_3 = False
        neighbor_vertices = set()
        for edge_connection in mesh.vertex_to_edges[vertex]:
            other_vertex = mesh.edges[
                edge_connection.edge_index, 1 - edge_connection.index_in_edge
            ]
            if len(mesh.vertex_to_edges[other_vertex]) == 3:
                neighbor_has_degree_3 = True
                break

            neighbor_vertices.add(other_vertex)
        return neighbor_vertices, neighbor_has_degree_3

    # The mesh will remain manifold as long as there are exactly two shared
    # neighbors.
    vertex_a_neighbors, a_neighbor_invalid = get_neighbor_vertices(vertex_a)
    vertex_b_neighbors, b_neighbor_invalid = get_neighbor_vertices(vertex_b)
    return (
        len(vertex_a_neighbors & vertex_b_neighbors) == 2
        and not a_neighbor_invalid
        and not b_neighbor_invalid
    )


def collapse_side(mesh: Mesh, edge_key: int, side: int, snapshot: CollapseSnapshot):
    """Collapse the triangle to the specified side of an edge that's being
    collapsed. Update edge_mask to exclude one of the edges that becomes
    overlapping as a result of the collapse. Calling this alone will leave
    the mesh in an inconsistent state.

    Returns the key for the edge that's masked off.
    """

    # The triangle that's being collapsed has three edges: the one that's
    # being collapsed, its even neighbor (side is 0 or 2) and its odd
    # neighbor (side is 1 or 3). After the collapse, the even neighbor
    # remains, while the odd neighbor is masked off.
    assert side == 0 or side == 2
    (
        even_neighbor_key,
        odd_neighbor_key,
        even_lookup,
        _,
        _,
        odd_lookup_opposite,
        _,
        odd_neighbor_neighbors,
    ) = get_face_neighborhood(mesh, edge_key, side)

    # Define functions to get the even and odd sides within a side group.
    get_even_side = lambda x: x - x % 2
    get_odd_side = lambda x: get_even_side(x) + 1

    # Rewrite the even neighbor's information to make the collapse happen.
    # First, replace the even neighbor's even neighbor (the one within the
    # collapsing triangle) with the odd neighbor's even neighbor (the one
    # outside the collapsing triangle).
    redirect_edges(
        mesh,
        even_neighbor_key,
        get_even_side(even_lookup),
        odd_neighbor_neighbors[0],
        mesh.edge_lookup[odd_neighbor_key, odd_lookup_opposite],
    )

    # Next, do the same for the even neighbor's odd neighbor.
    redirect_edges(
        mesh,
        even_neighbor_key,
        get_odd_side(even_lookup),
        odd_neighbor_neighbors[1],
        mesh.edge_lookup[odd_neighbor_key, odd_lookup_opposite + 1],
    )

    # Reparent the collapsed edge and the odd neighbor to the even neighbor.
    snapshot.reparent(odd_neighbor_key, even_neighbor_key)
    snapshot.reparent(edge_key, even_neighbor_key)

    # Mask off the odd neighbor.
    mesh.edge_mask[odd_neighbor_key] = False
    return odd_neighbor_key


def redirect_edges(
    mesh: Mesh,
    edge_key_a: int,
    b_side_for_a: int,
    edge_key_b: int,
    a_side_for_b: int,
):
    """Update edge_to_neighbors and edge_lookup so that edges A and B point
    towards each other. Calling this alone will leave the mesh in an
    inconsistent state.
    """
    mesh.edge_to_neighbors[edge_key_a, b_side_for_a] = edge_key_b
    mesh.edge_to_neighbors[edge_key_b, a_side_for_b] = edge_key_a
    mesh.edge_lookup[edge_key_a, b_side_for_a] = a_side_for_b
    mesh.edge_lookup[edge_key_b, a_side_for_b] = b_side_for_a


def get_face_neighborhood(mesh: Mesh, edge_key: int, side: int):
    """Gather information needed for edge collapse. In a watertight mesh,
    each triangular face has three adjacent faces.
    """
    assert side == 0 or side == 2

    # Find the keys for the neighbors on the specified sides.
    # Together, the original edge and its neighbors form a triangle.
    even_neighbor_key = mesh.edge_to_neighbors[edge_key, side]
    odd_neighbor_key = mesh.edge_to_neighbors[edge_key, side + 1]

    # Use edge_lookup to determine what side the original edge is on from
    # its neighbors' perspectives.
    even_lookup = mesh.edge_lookup[edge_key, side]
    odd_lookup = mesh.edge_lookup[edge_key, side + 1]

    # Define a lambda that gets the side index for the even neighbor that's
    # opposite an edge. Recall that an edge's 4 neighbors have sides 0
    # through 3, where (0, 1) and (2, 3) are on opposite sides of the edge.
    get_other_side = lambda x: 0 if x >= 2 else 2
    even_lookup_opposite = get_other_side(even_lookup)
    odd_lookup_opposite = get_other_side(odd_lookup)

    # For the original edge's neighbors on the specified side, get the other
    # two neighbors.
    even_neighbor_neighbors = (
        mesh.edge_to_neighbors[even_neighbor_key, even_lookup_opposite],
        mesh.edge_to_neighbors[even_neighbor_key, even_lookup_opposite + 1],
    )
    odd_neighbor_neighbors = (
        mesh.edge_to_neighbors[odd_neighbor_key, odd_lookup_opposite],
        mesh.edge_to_neighbors[odd_neighbor_key, odd_lookup_opposite + 1],
    )

    return (
        even_neighbor_key,
        odd_neighbor_key,
        even_lookup,
        odd_lookup,
        even_lookup_opposite,
        odd_lookup_opposite,
        even_neighbor_neighbors,
        odd_neighbor_neighbors,
    )