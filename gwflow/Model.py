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
            pickle.dump({'confined': self.confined,
                         'parameters': self.parameters,
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
        
        
class PickledModel(DefaultModel):
    def __init__(self, filename):
            
        with open(filename, 'rb') as f:
            pickle_dict = pickle.load(f)
            
        # Internalise solver parameters and initialise the solver
        self.confined = pickle_dict['confined']
        self.parameters = pickle_dict['parameters']
        
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
        
        self.conductivity = pickle_dict['conductivity']
        
        if self.confined:
            self.bottom_bounding_surface = None
        else:
            # Set the bottom bounding surface.
            self.bottom_bounding_surface = self.bottom_bounding_surface = pickle_dict['bottom_bounding_surface']
            self.solver.set_bottom_bounding_surface(self.bottom_bounding_surface.random_field)
        
class ReducedModel(DefaultModel):
    def __init__(self, fine_model, resolution):
            
        # Internalise solver parameters and initialise the solver
        self.confined = fine_model.confined
        self.parameters = fine_model.parameters.copy()
        self.parameters['resolution'] = resolution
        
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
        project_eigenpairs('conductivity', fine_model, self)
        
        if self.confined:
            self.bottom_bounding_surface = None
        else:
            # Set the bottom bounding surface.
            self.bottom_bounding_surface = self.parameters['bbs_kernel'](self.coords, self.parameters['bbs_lambda'])
            project_eigenpairs('bottom_bounding_surface', fine_model, self)
            self.bottom_bounding_surface.generate(self.parameters['bbs_coefficients'], 
                                                  self.parameters['bbs_mean'],
                                                  self.parameters['bbs_stdev'])
            self.solver.set_bottom_bounding_surface(self.bottom_bounding_surface.random_field)
        
    def plot_conductivity_eigenmode(self, index):
        
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
    
    if field_type == 'bottom_bounding_surface':
        fine_field = fine_model.bottom_bounding_surface
        coarse_field = coarse_model.bottom_bounding_surface
    elif field_type == 'conductivity':
        fine_field = fine_model.conductivity
        coarse_field = coarse_model.conductivity

    coarse_field.mkl = fine_field.mkl
    coarse_field.eigenvalues = fine_field.eigenvalues[:]
    
    coarse_field.eigenvectors = np.zeros((coarse_model.coords.shape[0], coarse_field.mkl))
    for i in range(coarse_field.mkl):
        phi_fine = fn.Function(fine_model.solver.V)
        phi_fine.vector()[:] = fine_field.eigenvectors[:, i]
        phi_coarse = fn.project(phi_fine, coarse_model.solver.V)
        coarse_field.eigenvectors[:, i] = phi_coarse.vector()[:]

