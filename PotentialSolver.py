import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from BoundaryClasses import BoundaryTypes


class PotentialSolver:
    """Solves the potentials at grid points given charge densities and boundary conditions. Uses the discretized
    Poisson equation (Laplacian(potential) = - (charge density) / (eps 0)"""
    def __init__(self, num_x_nodes, num_y_nodes, delta_x, delta_y, eps_0, boundary_conditions=[]):
        """
        :param num_x_nodes: num of nodes in x direction
        :param num_y_nodes: num of nodes in y direction
        :param delta_x: cell size in x direction
        :param delta_y: cell size in y direction
        :param eps_0: permittivity of free space
        :param boundary_conditions: list of BoundaryCondition objects
        """
        # TODO: Add iterative solver because direct solving via LU decomposition requires an excess of memory
        # TODO: Not solving potential correctly for surface charges

        self.boundary_conditions = boundary_conditions
        self.eps_0 = eps_0
        self.num_x_nodes = num_x_nodes
        self.num_y_nodes = num_y_nodes
        self.delta_y = delta_x
        self.delta_x = delta_y

        # here we set up the matrix, A, that is used to solve for the potentials
        # the RHS column vector is flattened in this order
        # (0, 0) (0, 1) (0, 2) ... (0, num_y_nodes) (1, 1) (1, 2) ... (num_xnodes, num_ynodes)

        # TODO: Make this matrix sparse initially
        self.size = num_x_nodes * num_y_nodes
        self.A = np.zeros((self.size, self.size))

        # apply the five point stencil
        for i, row in enumerate(self.A):
            row[i] = -2 / self.delta_x**2 - 2 / self.delta_y**2
            if i + 1 < self.size and not (i + 1) % self.num_y_nodes == 0:
                row[i+1] = 1 / self.delta_y**2
            if i - 1 >= 0 and not i % self.num_y_nodes == 0:
                row[i-1] = 1 / self.delta_y**2
            if i + self.num_y_nodes < self.size:
                row[i+self.num_y_nodes] = 1 / self.delta_x**2
            if i - self.num_y_nodes >= 0:
                row[i-self.num_y_nodes] = 1 / self.delta_x**2

        # apply boundary conditions
        for boundary_condition in boundary_conditions:
            for node_position in boundary_condition.positions.T:
                # find and reset the row that corresponds with this node coordinate
                row_num = (node_position[1] + node_position[0]*num_y_nodes)
                self.A[row_num] = np.zeros((1, self.size))

                if boundary_condition.type == BoundaryTypes.DIRICHLET:
                    self.A[row_num, row_num] = 1

                elif boundary_condition.type == BoundaryTypes.NEUMANN:
                    # on the boundaries we must do a forward / backward difference depending on the direction of
                    # the Neumann condition

                    if boundary_condition.neumann_direction == 0:  # in x dir
                        if node_position[0] == 0:  # LHS boundary
                            self.A[row_num, row_num] = -1 / self.delta_x
                            self.A[row_num, row_num + self.num_y_nodes] = 1 / self.delta_x
                        elif node_position[0] == self.num_x_nodes-1:  # RHS boundary
                            self.A[row_num, row_num] = 1 / self.delta_x
                            self.A[row_num, row_num - self.num_y_nodes] = -1 / self.delta_x
                        else:  # not on a boundary so we can do a central difference
                            self.A[row_num, row_num + self.num_y_nodes] = 1 / 2 / self.delta_x
                            self.A[row_num, row_num - self.num_y_nodes] = - 1 / 2 / self.delta_x

                    elif boundary_condition.neumann_direction == 1:  # in y dir
                        if node_position[1] == 0:  # bottom boundary
                            self.A[row_num, row_num] = -1 / self.delta_y
                            self.A[row_num, row_num + 1] = 1 / self.delta_y
                        elif node_position[1] == self.num_y_nodes-1:  # upper boundary
                            self.A[row_num, row_num] = 1 / self.delta_y
                            self.A[row_num, row_num - 1] = -1 / self.delta_y
                        else:  # not on a boundary so we can do a central difference
                            self.A[row_num, row_num + 1] = 1 / 2 / self.delta_y
                            self.A[row_num, row_num - 1] = - 1 / 2 / self.delta_y

        # for efficiency reasons, we solve this system using sparse LU decomposition
        self.A = scipy.sparse.csc_matrix(self.A)
        self.LU = scipy.sparse.linalg.splu(self.A)

    def solve_potentials(self, grid_charge_densities):
        """
        Solves the potentials at grid nodes
        :param grid_charge_densities: Array of charge densities at each node of the grid
        :return: returns array of potentials of equal dimensions to grid_charge_densities
        """

        # set up right hand vector, apply bcs
        p = -grid_charge_densities / self.eps_0
        for boundary_condition in self.boundary_conditions:
            p[boundary_condition.positions[0, :], boundary_condition.positions[1, :]] = boundary_condition.values

        # solve the system
        B = np.ndarray.flatten(p)
        X = self.LU.solve(B)

        # reshape and return
        grid_potentials = np.reshape(X, (self.num_x_nodes, self.num_y_nodes))
        return grid_potentials