class EdgeConnection:
    """This stores information about how a vertex is connected to an edge."""

    # The index of the edge the vertex is connected to.
    edge_index: int

    # The index of the vertex within the edge (i.e. 0 or 1).
    index_in_edge: int

    def __init__(self, edge_index: int, index_in_edge: int) -> None:
        assert edge_index >= 0
        assert index_in_edge == 0 or index_in_edge == 1
        self.edge_index = edge_index
        self.index_in_edge = index_in_edge

    def __hash__(self) -> int:
        return (self.edge_index, self.index_in_edge).__hash__()

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, EdgeConnection):
            return False
        return self.edge_index == o.edge_index and self.index_in_edge == o.index_in_edge