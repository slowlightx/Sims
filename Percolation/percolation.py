import numpy as np
import matplotlib.pyplot as plt


class DisjointSet:
    def __init__(self, n):
        self.nodes = [i for i in range(n)]
        self.father_nodes = [i for i in range(n)]
        self.rank = [1]*n

    def find_root(self, node):
        root_tmp = self.father_nodes[node]
        while root_tmp != node:
            node = root_tmp
            root_tmp = self.father_nodes[node]
        return root_tmp

    def is_union(self, node1, node2):
        return self.find_root(node1) == self.find_root(node2)

    def union_by_rank(self, node1, node2):
        if node1 is None or node2 is None:
            return
        root1 = self.find_root(node1)
        root2 = self.find_root(node2)
        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                self.father_nodes[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.father_nodes[root2] = root1
            else:
                self.father_nodes[root2] = root1
                self.rank[root1] += 1


class Percolation:
    ''' 2-Dimension Percolation '''
    def __init__(self, n):
        self.map = np.ones((n, n))
        self.connected_set = DisjointSet(n*n+2) # n*n and n*n+1 are respectively imaginary top and bottom
        self.n_visited = 0
        self.n = n
        self.dimension = 2

    def get_index(self, row, col):
        return row*self.n + col

    def visit(self, row, col):
        if not self.is_visited(row, col):
            self.n_visited += 1
            self.map[row, col] = 0

    def is_visited(self, row, col):
        return self.map[row, col] == 0

    def visit_neighbors(self, row, col):
        if self.is_visited(row, col):
            index = self.get_index(row, col)
            if row == 0:
                self.connected_set.union_by_rank(index, self.n*self.n)
            if row == self.n - 1:
                self.connected_set.union_by_rank(index, self.n*self.n+1)
            # adjacent_0 = [index-self.n, index+1, index+self.n, index-1]
            adjacent_0 = [[row-1, col], [row, col+1], [row+1, col], [row, col-1]]
            adjacent = [x for x in adjacent_0 if 0 <= x[0] < self.n and 0 <= x[1] < self.n]
            for x in adjacent:
                if self.is_visited(x[0], x[1]):
                    self.connected_set.union_by_rank(index, self.get_index(x[0], x[1]))

    def is_percolated(self):
        return self.connected_set.is_union(self.n*self.n, self.n*self.n+1)


class PercolationProcess:
    def __init__(self, n):
        self.lattice = Percolation(n)
        self.n_critical = 0

    # grow like random walks - another way where previous nodes can be visited twice or more times
    # def percolate(self):

    def get_critical_point(self):
        perm = np.random.permutation(self.lattice.n * self.lattice.n)
        for i in range(len(perm)):
            index = perm[i]
            row = index // self.lattice.n
            col = index % self.lattice.n
            self.lattice.visit(row, col)
            self.lattice.visit_neighbors(row, col)
            self.n_critical += 1
            if self.lattice.is_percolated():
                break
        return self.n_critical / (self.lattice.n * self.lattice.n)

    def percolate(self):
        l = self.lattice.n
        perm = np.random.permutation(l*l)
        for i in range(len(perm)):
            index = perm[i]
            row = index // l
            col = index % l
            self.lattice.visit(row, col)
            self.lattice.visit_neighbors(row, col)
            # every n/10 iterations plot a figure u
            if i % 16 == 0:
                figure = np.copy(self.lattice.map)
                for row in range(l):
                    for col in range(l):
                        if self.lattice.connected_set.is_union(row*l + col, l*l) \
                                or self.lattice.connected_set.is_union(row*l + col, l*l+1):
                            if self.lattice.is_percolated():
                                figure[row, col] = 0.7
                            else:
                                figure[row, col] = 0.3
                fig = plt.figure(1)
                pos = 240 + (i // 16) + 1
                ax = fig.add_subplot(pos)
                ax.imshow(figure, cmap='Blues', vmin=0, vmax=1)
        plt.show()

        return

    # def reset(self):


if __name__ == "__main__":

    n_l = 10
    n_iterations = 100

    probability_critical = 0

    i = 0
    while i < n_iterations:
        p = PercolationProcess(n_l)
        if i == 0:
            p.percolate()
        probability_critical += p.get_critical_point()
        i += 1

    print("The critical probability is")
    print(probability_critical/n_iterations)
