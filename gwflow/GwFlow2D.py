import numpy as np
import matplotlib.pyplot as plt

from fenics import *

from .Utils import BaseProblem, PointSourceProblem, CustomSolver

class GwFlowSolver:
    '''
    GwFlowSolver solves the groundwater flow problem.
    Initialise with folder name comtaining a FEniCS mesh with name mesh.xml and FEM degree.
    Solve by passing a FEniCS Function with conductivity scalars.
    '''
    
    def __init__(self, parameters):
        
        # Set the FEM degree for creating Expressions and FunctionSpaces
        self.parameters = parameters
        self.fem_degree = 1
        
        self.mesh = RectangleMesh(Point(0,0), Point(*self.parameters['extent']), *self.parameters['resolution'])
        
    def refine_mesh(self):
        
        distance = 5*self.mesh.rmax()
        
        for location in self.parameters['pumping_locations']:
            cell_markers = MeshFunction("bool", self.mesh, 2)
            cell_markers.set_all(False)
            for cell in cells(self.mesh):
                if cell.midpoint().distance(Point(location)) < distance:
                    cell_markers[cell] = True
            self.mesh = refine(self.mesh, cell_markers)
                
    def setup(self):
        
        self.boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim() - 1)
        self.boundaries.set_all(2)
        
        h_in = CompiledSubDomain('on_boundary && near(x[0], 0, tol)', tol=DOLFIN_EPS)
        h_in.mark(self.boundaries, 1)
        
        h_out = CompiledSubDomain(f'on_boundary && near(x[0], {self.parameters["extent"][0]}, tol)', tol=DOLFIN_EPS)
        h_out.mark(self.boundaries, 0)

        # Impose the subdomain numbering on external boundary ds.
        self.ds = ds(subdomain_data=self.boundaries)
        
        # Set up a FunctionSpace for the pressure solution and a VectorFunctionSpace for the flow.
        self.V = FunctionSpace(self.mesh, 'CG', self.fem_degree)
        #self.VV = FunctionSpace(self.mesh, 'DG', 0)
        self.Q = VectorFunctionSpace(self.mesh, 'CG', self.fem_degree)
        self.n = FacetNormal(self.mesh)
        
        # Set boundary conditions
        # No flow across boundaries marked with "0"
        self.q_0 = Constant(0.0)
        # Fixed head boundaries
        self.bcs = [DirichletBC(self.V, 0, self.boundaries, 0),
                    DirichletBC(self.V, 1, self.boundaries, 1)]
        
        # Set the conductivity field.
        self.K = Function(self.V, name='K')
        
        # Initialise the testing functions.
        self.v = TestFunction(self.V)
        
        # Set sources and sinks.
        if 'sources_and_sinks' in self.parameters.keys():
            self.G = Expression(self.parameters['sources_and_sinks'], degree=self.fem_degree)
        else:
            self.G = Expression('0', degree=self.fem_degree)
        
        if 'pumping_rates' in self.parameters.keys():
            self.pumping = True
            self.set_pumping()
        else:
            self.pumping = False
        
    def set_pumping(self):
        self.pss = []
        for Q_i, x_i in zip(self.parameters['pumping_rates'], self.parameters['pumping_locations']):
            if isinstance(self, Unconfined):
                Q_i = -Q_i
            self.pss.append(PointSource(self.V, Point(x_i), Q_i))
        
    def get_head(self, datapoints):
        
        # Return data from a set of points.
        # Columns are dimensions, rows are datapoints.
        head = np.zeros(len(datapoints))
        for d, datapoint in enumerate(datapoints):
            head[d] = self.h(*datapoint)
        return head
        
    def compute_flux(self):
        # Compute the flow on the entire domain.
        self.q = project(-self.K*grad(self.h), self.Q)
        
    def get_flux(self, datapoints):
        # Return data from a set of points.
        # Columns are dimensions, rows are datapoints.
        flux = np.zeros((len(datapoints), 2))
        for d, datapoint in enumerate(datapoints):
            flux[d,:] = self.q(*datapoint)
        return flux
        
    def get_outflux(self):
        
        # Compute the outflow over the constant head boundary.
        return assemble(inner(-self.K*grad(self.h), self.n)*self.ds(0))


class Confined(GwFlowSolver):
    
    def assemble(self):
        
        self.u = TrialFunction(self.V)
        
        # Assemble the system.
        a = inner(grad(self.v), self.K*grad(self.u))*dx
        L = self.v*self.G*dx - self.v*self.q_0*self.ds(2)
        
        self._A = PETScMatrix()
        self._b = PETScVector()
        self._assembler = SystemAssembler(a, L, self.bcs)
        self._solver = PETScKrylovSolver('gmres', 'ilu')
        self._solver.set_operator(self._A)
        
    def solve(self, conductivity, **kwargs):
        
        if not hasattr(self, '_solver'):
            self.assemble()
        
        # Sove using some FEniCS function with conductivity scalars as input.
        self.K.assign(conductivity)
        self.h = Function(self.V, name='h')
        self._assembler.assemble(self._A, self._b)
        
        if self.pumping:
            for ps in self.pss:
                ps.apply(self._b)
        
        try:
            self._solver.solve(self.h.vector(), self._b)
        except RuntimeError:
            self.h.vector()[:] = 0
        return self.h
        
        
class Unconfined(GwFlowSolver):
    
    def set_bottom_bounding_surface(self, bottom_bounding_surface):
        # Set bottom bounding surface
        self.f_B = Function(self.V)
        self.f_B.vector()[:] = bottom_bounding_surface
        
    def assemble(self):
        
        self.u = Function(self.V)
        self.du = TrialFunction(self.V)
        
        self.F = inner(grad(self.v), (self.u-self.f_B)*self.K*grad(self.u))*dx - self.v*self.G*dx + self.v*(self.u-self.f_B)*self.q_0*self.ds(2)
        self.J = derivative(self.F, self.u, self.du)
        
        self._solver = None
        
    def solve(self, conductivity, **kwargs):
        
        if not hasattr(self, '_solver'):
            self.assemble()
        
        self.K.assign(conductivity)
        self.h = Function(self.V, name='h')
        
        if self.pumping:
            problem = PointSourceProblem(self.J, self.F, self.bcs, self.pss)
        else:
            problem = BaseProblem(self.J, self.F, self.bcs)
        
        solver = CustomSolver(self.mesh)
        solver.solve(problem, self.u.vector())
        
        #solve(F == 0, self.u, self.bcs, J=J, solver_parameters=kwargs['solver_parameters'])
        self.h.assign(self.u)
        
        self.B = project(self.h - self.f_B, self.V)
    
    def compute_flux(self):
        # Compute the flow on the entire domain.
        self.q = project(-(self.h - self.f_B)*self.K*grad(self.h), self.Q)
        
    def get_outflux(self):
        # Compute the outflow over the constant head boundary.
        return assemble(dot(-(self.h - self.f_B)*self.K*grad(self.h), self.n)*self.ds(0))

