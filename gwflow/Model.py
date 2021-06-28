import pickle
import numpy as np
import matplotlib.pyplot as plt

import fenics as fn
fn.parameters['allow_extrapolation'] = True

from .RandomProcess import *
from .GwFlow2D import Confined, Unconfined

class DefaultModel:
    def __init__(self, parameters, confined=True):
        
        self.parameters = parameters
        self.confined = confined
        
        if self.confined:
            self.solver = Confined(self.parameters)
        else:
            self.solver = Unconfined(self.parameters)
            
        if 'pumping_locations' in self.parameters.keys():
            self.solver.refine_mesh()
            
        self.solver.setup()
            
        dof_coords = self.solver.V.tabulate_dof_coordinates().reshape((-1, 2))
        dof_indices = self.solver.V.dofmap().dofs()
        self.coords = dof_coords[dof_indices, :]
        
        self.conductivity = self.parameters['conductivity_kernel'](self.coords, self.parameters['conductivity_lambda'])
        self.conductivity.compute_eigenpairs(self.parameters['conductivity_mkl'])
        
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
            
            
    def solve(self, coefficients=None):
        
        self.coefficients = coefficients
        self.conductivity.generate(self.coefficients,
                                   self.parameters['conductivity_mean'],
                                   self.parameters['conductivity_stdev'])
        self.K = fn.Function(self.solver.V)
        self.K.vector()[:] = np.exp(self.conductivity.random_field)
        
        self.solver.solve(self.K, solver_parameters=self.parameters['solver_parameters'])
        
        self.solver.compute_flux()
        
    def get_data(self, datapoints):
        head = self.solver.get_head(datapoints)
        flux = self.solver.get_flux(datapoints)
        return np.hstack((head, np.linalg.norm(flux, axis=1)))
        
    def pickle(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'parameters': self.parameters,
                         'bottom_bounding_surface': self.bottom_bounding_surface,
                         'conductivity': self.conductivity}, f)
                         
    def plot_mesh(self):
        plt.figure(figsize=(10,4))
        mesh_plot = fn.plot(self.solver.mesh)
        plt.show()
        
    def plot_bottom_bounding_surface(self):
        plt.figure(figsize=(10,4))
        bbs_plot = fn.plot(self.solver.f_B)
        plt.colorbar(bbs_plot)
        plt.show()
        
    def plot_conductivity_eigenmode(self, index):
        
        phi = fn.Function(self.solver.V)
        phi.vector()[:] = self.conductivity.eigenvectors[:, index]
        
        plt.figure(figsize=(10,4))
        phi_plot = fn.plot(phi)
        plt.colorbar(phi_plot)
        plt.show()
        

    def plot_conductivity(self, log=True):
        
        if log == True:
            K = fn.ln(self.K)
        else:
            K = self.K
        
        plt.figure(figsize=(10,4))
        K_plot = fn.plot(K)
        plt.colorbar(K_plot)
        plt.show()
        
    def plot_head(self):
        plt.figure(figsize=(10,4))
        h_plot = fn.plot(self.solver.h)
        plt.colorbar(h_plot)
        plt.show()
        
    def plot_flux(self):
        plt.figure(figsize=(10,4))
        q_plot = fn.plot(self.solver.q)
        plt.colorbar(q_plot)
        plt.show()
        
        
#class PickledGwModel(GwModel):
#    def __init__(self, filename):
#            
#        with open(filename, 'rb') as f:
#            pickle_dict = pickle.load(f)
#            
#        # Internalise solver parameters and initialise the solver
#        self.gw_parameters = pickle_dict['gw_parameters']
#        
#        if self.gw_parameters['solver'] == 'confined':
#            self.gw_solver = Confined(self.gw_parameters['size'], self.gw_parameters['resolution'])
#            self.solver_parameters = None
#        
#        elif self.gw_parameters['solver'] == 'unconfined':
#            self.gw_solver = Unconfined(self.gw_parameters['size'], self.gw_parameters['resolution'])
#            self.solver_parameters = self.gw_parameters['solver_parameters']
#        
#        else:
#            raise TypeError('Solver type not understood')
#        
#        dof_coords = self.gw_solver.V.tabulate_dof_coordinates().reshape((-1, 2))
#        dof_indices = self.gw_solver.V.dofmap().dofs()
#        self.coords = dof_coords[dof_indices, :]
#        
#        # Set the forcing term parameters.
#        self.gw_solver.set_sources_and_sinks(self.gw_parameters['sources_and_sinks'])
#
#        if self.gw_parameters['solver'] == 'confined':
#            self.bottom_bounding_surface = None
#        
#        elif self.gw_parameters['solver'] == 'unconfined':        
#            # Set the bottom bounding surface.
#            self.bottom_bounding_surface = pickle_dict['bottom_bounding_surface']
#            self.bottom_bounding_surface.mkl = self.gw_parameters['mkl_bbs']
#            self.gw_solver.set_bottom_bounding_surface(self.bottom_bounding_surface.random_field)
#        
#        self.conductivity_parameters = pickle_dict['conductivity_parameters']
#        
#        self.conductivity = pickle_dict['conductivity']
#        self.conductivity.mkl = self.conductivity_parameters['mkl']
#        
#class ReducedGwModel(GwModel):
#    def __init__(self, fine_model, resolution):
#            
#        # Internalise solver parameters and initialise the solver
#        self.gw_parameters = fine_model.gw_parameters.copy()
#        self.gw_parameters['resolution'] = resolution
#        
#        if self.gw_parameters['solver'] == 'confined':
#            self.gw_solver = Confined(self.gw_parameters['size'], self.gw_parameters['resolution'])
#            self.solver_parameters = None
#        
#        elif self.gw_parameters['solver'] == 'unconfined':
#            self.gw_solver = Unconfined(self.gw_parameters['size'], self.gw_parameters['resolution'])
#            self.solver_parameters = self.gw_parameters['solver_parameters']
#        
#        else:
#            raise TypeError('Solver type not understood')
#        
#        dof_coords = self.gw_solver.V.tabulate_dof_coordinates().reshape((-1, 2))
#        dof_indices = self.gw_solver.V.dofmap().dofs()
#        self.coords = dof_coords[dof_indices, :]
#        
#        # Set the forcing term parameters.
#        self.gw_solver.set_sources_and_sinks(self.gw_parameters['sources_and_sinks'])
#        
#        
#        if self.gw_parameters['solver'] == 'confined':
#            self.bottom_bounding_surface = None
#        
#        elif self.gw_parameters['solver'] == 'unconfined':   
#            # Set the bottom bounding surface.
#            self.bottom_bounding_surface = Matern52(self.coords,
#                                                    self.gw_parameters['lambda_bbs'])
#            self.bottom_bounding_surface.mkl = self.gw_parameters['mkl_bbs']
#            
#            project_eigenpairs('bottom_bounding_surface', fine_model, self)
#            
#            self.bottom_bounding_surface.generate(self.gw_parameters['coefficients_bbs'], 
#                                                  self.gw_parameters['mean_bbs'],
#                                                  self.gw_parameters['stdev_bbs'])
#            self.gw_solver.set_bottom_bounding_surface(self.bottom_bounding_surface.random_field)
#        
#        self.conductivity_parameters = fine_model.conductivity_parameters.copy()
#        
#        self.conductivity = self.conductivity_parameters['kernel'](
#                                self.coords,
#                                self.conductivity_parameters['lambda'])
#        self.conductivity.mkl = self.conductivity_parameters['mkl']
#        
#        project_eigenpairs('conductivity', fine_model, self)
#        
#        #self.solver_parameters = {'newton_solver': {'relative_tolerance': 1e-6,
#        #                                            'absolute_tolerance': 1e-6,
#        #                                            'maximum_iterations': 10,
#        #                                            'relaxation_parameter': 1.0}}
#        
#    def plot_conductivity_eigenmode(self, index):
#        
#        phi = fn.Function(self.gw_solver.V)
#        phi.vector()[:] = 1*self.conductivity.eigenvectors[:, index]
#        
#        plt.figure(figsize=(10,4))
#        phi_plot = fn.plot(phi)
#        plt.colorbar(phi_plot)
#        plt.show()
#
#def project_eigenpairs(field_type, fine_model, coarse_model):
#   
#    """
#    Projects eigenpairs from a fine model to a coarse model.
#    """
#    
#    if field_type == 'bottom_bounding_surface':
#        fine_field = fine_model.bottom_bounding_surface
#        coarse_field = coarse_model.bottom_bounding_surface
#    elif field_type == 'conductivity':
#        fine_field = fine_model.conductivity
#        coarse_field = coarse_model.conductivity
#
#    coarse_field.eigenvalues = fine_field.eigenvalues[:]
#    
#    coarse_field.eigenvectors = np.zeros((coarse_model.coords.shape[0], fine_field.eigenvectors.shape[1]))
#    for i in range(len(coarse_field.eigenvalues)):
#        phi_fine = fn.Function(fine_model.gw_solver.V)
#        phi_fine.vector()[:] = fine_field.eigenvectors[:, i]
#        phi_coarse = fn.project(phi_fine, coarse_model.gw_solver.V)
#        coarse_field.eigenvectors[:, i] = phi_coarse.vector()[:]
#
