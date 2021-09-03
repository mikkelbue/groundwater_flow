import pickle
import numpy as np
import matplotlib.pyplot as plt

import fenics as fn
fn.parameters['allow_extrapolation'] = True

from .RandomProcess import *
from .GwFlow2D import Confined, Unconfined

class DefaultModel:
    '''
    DefaultModel implements a steady state groundwater flow model.
    It can be confined (default) or unconfined.
    
    The problem is solved on a rectangular grid, with fixed head of 
    1 and 0 at left and right boundaries, respectively. Remaining 
    boundaries are no-flux.
    
    Conductivity is modelled as a Gaussian random field, implemented
    using a KL decomposition.
    
    Body forces are implemented as a fenics Expression, given in the
    parameters as a string with key 'sources_and_sinks'.
    
    An arbitrary number of wells can be added, given in the parameters 
    as two lists with keys 'pumping_rates' and 'pumping_locations'.
    These will be added to the problem as fenics.PointSource.
    
    Please refer to the example notebook for details of use.
    '''
    
    def __init__(self, parameters, confined=True):
        
        # internalise parameters and solver type.
        self.parameters = parameters
        self.confined = confined
        
        # set up the solver, according to the type.
        if self.confined:
            self.solver = Confined(self.parameters)
        else:
            self.solver = Unconfined(self.parameters)
            
        # refine mesh at pumping locations, if there is any pumping.
        if 'pumping_locations' in self.parameters.keys():
            self.solver.refine_mesh()
            
        # set up the remaining solver elements (mostly fenics objects)
        self.solver.setup()
            
        # get the node coordinates.
        dof_coords = self.solver.V.tabulate_dof_coordinates().reshape((-1, 2))
        dof_indices = self.solver.V.dofmap().dofs()
        self.coords = dof_coords[dof_indices, :]
        
        # set up the conductivity.
        self.conductivity = self.parameters['conductivity_kernel'](self.coords, self.parameters['conductivity_lambda'])
        self.conductivity.compute_eigenpairs(self.parameters['conductivity_mkl'])
        
        # set up the bottom bounding surface, if this is an unconfined solver.
        if self.confined:
            self.bottom_bounding_surface = None
        else:
            # Set the bottom bounding surface.
            self.bottom_bounding_surface = self.parameters['bbs_kernel'](self.coords, self.parameters['bbs_lambda'])
            self.bottom_bounding_surface.compute_eigenpairs(self.parameters['bbs_mkl'])
            self.bottom_bounding_surface.generate(self.parameters['bbs_coefficients'], 
                                                  self.parameters['bbs_mean'],
                                                  self.parameters['bbs_stdev'])
            self.solver.set_bottom_bounding_surface(self.bottom_bounding_surface.random_field)
            
        if 'datapoints' in self.parameters.keys():
            self.datapoints = self.parameters['datapoints']
            
        if 'solver_parameters' not in self.parameters.keys():
            self.parameters['solver_parameters'] = None

    def __call__(self, coefficients):
        
        # solve given the coefficients.
        self.solve(coefficients)
        
        # return the data.
        return self.get_data(self.datapoints)
            
    def solve(self, coefficients=None):
        
        # solve the problem given some input KL coefficients.
        self.coefficients = coefficients
        
        # generate a random field realisation.
        self.conductivity.generate(self.coefficients,
                                   self.parameters['conductivity_mean'],
                                   self.parameters['conductivity_stdev'])
        # set the conductivity as a fenics.Function.
        self.K = fn.Function(self.solver.V)
        self.K.vector()[:] = np.exp(self.conductivity.random_field)
        
        # solve the problem.
        self.solver.solve(self.K, solver_parameters=self.parameters['solver_parameters'])
        
        # compute the flux.
        self.solver.compute_flux()
        
    def get_data(self, datapoints):
        # get the head and flux magnitude at each datapoint.
        head = self.solver.get_head(datapoints)
        flux = self.solver.get_flux(datapoints)
        return np.hstack((head, np.linalg.norm(flux, axis=1)))
        
    def pickle(self, filename):
        # pickle the model. fenics objects mostly do not pickle, so this
        # is a workaround for that.
        with open(filename, 'wb') as f:
            pickle.dump({'confined': self.confined,
                         'parameters': self.parameters,
                         'bottom_bounding_surface': self.bottom_bounding_surface,
                         'conductivity': self.conductivity}, f)
                         
    def plot_mesh(self):
        # plot the mesh.
        plt.figure(figsize=(10,4))
        mesh_plot = fn.plot(self.solver.mesh)
        plt.show()
        
    def plot_bottom_bounding_surface(self):
        if self.confined:
            print('Confined problem has no bottom bounding surface')
            return
        else:
            plt.figure(figsize=(10,4))
            bbs_plot = fn.plot(self.solver.f_B)
            plt.colorbar(bbs_plot)
            plt.show()
        
    def plot_conductivity_eigenmode(self, index):
        # plot an KL eigenmode given the index.
        phi = fn.Function(self.solver.V)
        phi.vector()[:] = self.conductivity.eigenvectors[:, index]
        
        plt.figure(figsize=(10,4))
        phi_plot = fn.plot(phi)
        plt.colorbar(phi_plot)
        plt.show()
        

    def plot_conductivity(self, log=True):
        # plot the conductivity field (log or nor)
        if log == True:
            K = fn.ln(self.K)
        else:
            K = self.K
        
        plt.figure(figsize=(10,4))
        K_plot = fn.plot(K)
        plt.colorbar(K_plot)
        plt.show()
        
    def plot_head(self):
        # plot the head.
        plt.figure(figsize=(10,4))
        h_plot = fn.plot(self.solver.h)
        plt.colorbar(h_plot)
        plt.show()
        
    def plot_flux(self):
        #plot the flux.
        plt.figure(figsize=(10,4))
        q_plot = fn.plot(self.solver.q)
        plt.colorbar(q_plot)
        plt.show()
        
        
class PickledModel(DefaultModel):
    def __init__(self, filename):
        
        # unpickle
        with open(filename, 'rb') as f:
            pickle_dict = pickle.load(f)
            
        # internalise parameters and solver type.
        self.confined = pickle_dict['confined']
        self.parameters = pickle_dict['parameters']
        
        # set up the solver, according to the type.
        if self.confined:
            self.solver = Confined(self.parameters)
        else:
            self.solver = Unconfined(self.parameters)
            
        if 'pumping_locations' in self.parameters.keys():
            self.solver.refine_mesh()
        
        # set up the remaining solver elements (mostly fenics objects)
        self.solver.setup()
        
        # get the node coordinates.
        dof_coords = self.solver.V.tabulate_dof_coordinates().reshape((-1, 2))
        dof_indices = self.solver.V.dofmap().dofs()
        self.coords = dof_coords[dof_indices, :]
        
        # get the pickled conductivity object.
        self.conductivity = pickle_dict['conductivity']
        
        # set up the bottom bounding surface, if this is an unconfined solver.
        if self.confined:
            self.bottom_bounding_surface = None
        else:
            # get the pickled bottom bounding surface.
            self.bottom_bounding_surface = self.bottom_bounding_surface = pickle_dict['bottom_bounding_surface']
            self.solver.set_bottom_bounding_surface(self.bottom_bounding_surface.random_field)
        
class ReducedModel(DefaultModel):
    def __init__(self, fine_model, resolution):
            
         # internalise parameters and solver type.
        self.confined = fine_model.confined
        self.parameters = fine_model.parameters.copy()
        
        # overwrite the resolution.
        self.parameters['resolution'] = resolution
        
        # set up the solver, according to the type.
        if self.confined:
            self.solver = Confined(self.parameters)
        else:
            self.solver = Unconfined(self.parameters)
        
        # refine mesh at pumping locations, if there is any pumping.
        if 'pumping_locations' in self.parameters.keys():
            self.solver.refine_mesh()
        
        # set up the remaining solver elements (mostly fenics objects)
        self.solver.setup()
        
        # get the node coordinates.
        dof_coords = self.solver.V.tabulate_dof_coordinates().reshape((-1, 2))
        dof_indices = self.solver.V.dofmap().dofs()
        self.coords = dof_coords[dof_indices, :]
        
        # set up the conductivity, but skip the eigenvalue decomposition.
        self.conductivity = self.parameters['conductivity_kernel'](self.coords, self.parameters['conductivity_lambda'])
        # instead, project the eigenpairs from the fine model.
        project_eigenpairs('conductivity', fine_model, self)
        
        # set up the bottom bounding surface, if this is an unconfined solver.
        if self.confined:
            self.bottom_bounding_surface = None
        else:
            # Set the bottom bounding surface, but skip the eigenvalue decomposition.
            self.bottom_bounding_surface = self.parameters['bbs_kernel'](self.coords, self.parameters['bbs_lambda'])
            # instead, project the eigenpairs from the fine model.
            project_eigenpairs('bottom_bounding_surface', fine_model, self)
            self.bottom_bounding_surface.generate(self.parameters['bbs_coefficients'], 
                                                  self.parameters['bbs_mean'],
                                                  self.parameters['bbs_stdev'])
            self.solver.set_bottom_bounding_surface(self.bottom_bounding_surface.random_field)
        
    def plot_conductivity_eigenmode(self, index):
        # plot an KL eigenmode given the index.
        # this overwrites the default method, since there is some strange bug.
        phi = fn.Function(self.gw_solver.V)
        phi.vector()[:] = 1*self.conductivity.eigenvectors[:, index]
        
        plt.figure(figsize=(10,4))
        phi_plot = fn.plot(phi)
        plt.colorbar(phi_plot)
        plt.show()

def project_eigenpairs(field_type, fine_model, coarse_model):
   
    """
    Projects eigenpairs from a fine model to a coarse model.
    """
    
    # pick the object accroding the field type.
    if field_type == 'bottom_bounding_surface':
        fine_field = fine_model.bottom_bounding_surface
        coarse_field = coarse_model.bottom_bounding_surface
    elif field_type == 'conductivity':
        fine_field = fine_model.conductivity
        coarse_field = coarse_model.conductivity

    # get the truncation length and the eigenvalues.
    coarse_field.mkl = fine_field.mkl
    coarse_field.eigenvalues = fine_field.eigenvalues[:]
    
    # project the eigenmodes into the coarse space.
    coarse_field.eigenvectors = np.zeros((coarse_model.coords.shape[0], coarse_field.mkl))
    for i in range(coarse_field.mkl):
        phi_fine = fn.Function(fine_model.solver.V)
        phi_fine.vector()[:] = fine_field.eigenvectors[:, i]
        phi_coarse = fn.project(phi_fine, coarse_model.solver.V)
        coarse_field.eigenvectors[:, i] = phi_coarse.vector()[:]

