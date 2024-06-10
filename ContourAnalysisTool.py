#===========================================================
#
# ContourAnalysisTool.py - A set of tools to analyse/process 2D images 
# to interactively and programmatically identify regions of interest.
#
# Copyright (C) 2024 Mark Hutchison
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
#===========================================================
#
# Dependencies:
#
# This script utilizes the following third-party Python packages:
# - numpy:        Essential for numerical computations.
# - matplotlib:   Comprehensive library for visualizations.
# - scikit-image: Collection of image processing algorithms.
# - scipy:        Library for scientific and technical computing.
# - ipywidgets:   Interactive HTML widgets for Jupyter notebooks.
# - IPython:      Rich architecture for interactive computing.
# - loess:        LOESS (Local Regression) smoothing for 2D data.
# - fuzzywuzzy:   Library for fuzzy string matching.
# - os:           Operating system-dependent functionality.
#
# Missing packages can be installed using pip:
# pip install numpy matplotlib scikit-image scipy ipywidgets IPython loess fuzzywuzzy os
#
#===========================================================


#***********************************************************
#===========================================================
#               Core Code Functionality
#===========================================================
#***********************************************************

import numpy as np
import matplotlib.pyplot as plt

#-------------------
# Utility Class
#-------------------
class Utilities:
    """
    A collection of utility functions for class independent tasks including rounding values, 
    creating widgets, and updating widget values.
    """

    @staticmethod
    def is_jupyter_notebook():
        """
        Check if the current environment is a Jupyter notebook.

        Returns:
            bool: True if running in a Jupyter notebook, False otherwise.
        """
        try:
            from IPython import get_ipython
            if 'IPKernelApp' not in get_ipython().config:
                return False
        except:
            return False
        return True

    @staticmethod
    def set_matplotlib_backend():
        """
        Set the matplotlib backend to 'widget' if running in a Jupyter notebook.

        This enables interactive plots in Jupyter notebooks.
        """
        if Utilities.is_jupyter_notebook():
            from IPython import get_ipython
            get_ipython().run_line_magic('matplotlib', 'widget')

    @staticmethod
    def round_to_significant(value, significant_figures):
        """
        Rounds a given value to the specified number of significant figures.
        
        Parameters:
        value (float): The value to be rounded.
        significant_figures (int): The number of significant figures to round to.

        Returns:
        tuple: A tuple containing the rounded value and the order of magnitude.
        """
        order_of_magnitude = int(np.floor(np.log10(np.abs(value))))
        rounded_value = np.around(value, decimals=-order_of_magnitude + (significant_figures - 1))
        return rounded_value, 10**order_of_magnitude

    @staticmethod
    def calculate_step_size(value):
        """
        Calculates the step size for a given value based on its order of magnitude.
        
        Parameters:
        value (float): The value to calculate the step size for.

        Returns:
        float: The calculated step size.
        """
        magnitude = int(np.floor(np.log10(abs(value))))
        return 10 ** magnitude

    @staticmethod
    def enforce_limits(change, text):
        """
        Enforces the minimum and maximum limits on float input widgets.
        
        Parameters:
        change (dict): The change dictionary containing the new value.
        text (widgets.BoundedFloatText): The text widget to enforce limits on.
        """
        if change.new < text.min:
            text.value = text.min
        elif change.new > text.max:
            text.value = text.max

    @staticmethod
    def create_float_text(description, value, min_value, max_value, step, width='200px', **kwargs):
        """
        Creates a text input widget for float values with observe functionality.
        
        Parameters:
        description (str): The description of the widget.
        value (float): The initial value of the widget.
        min_value (float): The minimum value allowed.
        max_value (float): The maximum value allowed.
        step (float): The step size for the widget.
        width (str): The width of the widget.

        Returns:
        widgets.BoundedFloatText: The created float text widget.
        """
        import ipywidgets as widgets

        text = widgets.BoundedFloatText(description=description, value=value, min=min_value, max=max_value, step=step, layout=widgets.Layout(width=width))
        text.observe(lambda change: Utilities.enforce_limits(change, text), names='value')
        return text

    @staticmethod
    def create_int_text(description, value, min_value, max_value, step=1, width='150px', **kwargs):
        """
        Creates a text input widget for integer values with observe functionality.
        
        Parameters:
        description (str): The description of the widget.
        value (int): The initial value of the widget.
        min_value (int): The minimum value allowed.
        max_value (int): The maximum value allowed.
        width (str): The width of the widget.
        **kwargs: Additional keyword arguments.

        Returns:
        widgets.BoundedIntText: The created integer text widget.
        """
        import ipywidgets as widgets

        text = widgets.BoundedIntText(description=description, value=value, min=min_value, max=max_value, step=step, layout=widgets.Layout(width=width))
        text.observe(lambda change: Utilities.enforce_limits(change, text), names='value')
        return text

    @staticmethod
    def create_dropdown(description, options, value):
        """
        Creates a dropdown widget.
        
        Parameters:
        description (str): The description of the dropdown.
        options (list): The list of options for the dropdown.
        value (str): The initial value of the dropdown.

        Returns:
        widgets.Dropdown: The created dropdown widget.
        """
        import ipywidgets as widgets

        return widgets.Dropdown(description=description, options=options, value=value)

    @staticmethod
    def update_min_value(min_input, max_input, image_min):
        """
        Updates the minimum value of a widget based on the image minimum value.
        
        Parameters:
        min_input (widgets.BoundedFloatText): The widget for the minimum value.
        max_input (widgets.BoundedFloatText): The widget for the maximum value.
        image_min (float): The minimum value based on the image data.
        """
        if min_input.value < image_min:
            min_input.value = image_min
        if min_input.value > max_input.value:
            min_input.value = max_input.value - min_input.step/100

    @staticmethod
    def update_max_value(min_input, max_input, image_max):
        """
        Updates the maximum value of a widget based on the image maximum value.
        
        Parameters:
        min_input (widgets.BoundedFloatText): The widget for the minimum value.
        max_input (widgets.BoundedFloatText): The widget for the maximum value.
        image_max (float): The maximum value based on the image data.
        """
        if max_input.value < min_input.value:
            max_input.value = min_input.value + max_input.step
        if max_input.value > image_max:
            max_input.value = image_max

    @staticmethod
    def update_step_size(min_input, max_input, change):
        """
        Updates the step size of a widget based on the changed value.
        
        Parameters:
        min_input (widgets.BoundedFloatText): The widget for the minimum value.
        max_input (widgets.BoundedFloatText): The widget for the maximum value.
        change (dict): The change dictionary containing the new value.
        """
        value = change['new']
        step_size = Utilities.calculate_step_size(value)
        if value <= 0:
            step_size = 10 ** (np.floor(np.log10(abs(value))) - 1)
        min_input.step = step_size
        max_input.step = step_size
#------------------- End Utilities


#-------------------
# Image Data
#-------------------
class ImageData:
    """
    A collection of variables relating to the image and its visualisation,
    including handling endianness, generating plot labels and parameters, 
    and initialising figures.
    """
    
    def __init__(self, image_raw, extent):
        """
        Initialize the ImageData object with raw image data and its extent.
        
        Parameters:
        image_raw (numpy.ndarray): The unmodified raw image supplied by the user.
        extent (list): The physical extent of the image in the format [xmin, xmax, ymin, ymax].
        """
        # Containers for the various images used/produced in the program
        self.image_raw            = self.fix_endianness(image_raw)
        self.image_smoothed       = None
        self.image_nobackground   = None
        self.estimated_background = None
        self.image_difference     = None

        # Containers for the selected background and colorbar
        self.background_cmap      = None
        self.cbar                 = None

        # Split the extent into x and y mins and maxes
        self.extent = extent
        self.xmin   = self.extent[0]
        self.xmax   = self.extent[1]
        self.ymin   = self.extent[2]
        self.ymax   = self.extent[3]

        # Create x and y axes (row major order)
        self.x = np.linspace(self.xmin, self.xmax, self.image_raw.shape[1])
        self.y = np.linspace(self.ymin, self.ymax, self.image_raw.shape[0])

        # 2D meshgrid of x and y
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Spacing between grid points in x and y
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        # Containers for the min and max of the various images
        self.image_min                = np.min(self.image_raw)
        self.image_max                = np.max(self.image_raw)
        self.image_min_smoothed       = None
        self.image_max_smoothed       = None
        self.image_min_nobackground   = None
        self.image_max_nobackground   = None

        # Arbitrary upper or lower limits for various parameters
        self.small_val = 1e-300
        self.large_val = 1e300
        self.large_int = 1000000
        self.rounded_min, self.rounded_min_magnitude = Utilities.round_to_significant(self.image_min, 1)

        # Set the min and max of the colorbar
        self.cbar_min, _ = Utilities.round_to_significant(self.image_min, 3)
        self.cbar_max, _ = Utilities.round_to_significant(self.image_max, 3)

        # Default number of contour levels used
        self.Nlevels     = 1000

        # Default colormap used
        self.cmapname    = 'viridis'

        # Make no assumptions about the units
        self.xy_units    = ''
        self.cbar_units  = ''

        # Plot labels
        self.xlabel      = '$x$'
        self.ylabel      = '$y$'
        self.clabel      = ''

        # Default line styles the Raw, Smoothed, and NoBackground contour sets
        self.line_style  = ['solid', 'dashed', 'dotted']

        # Default line colors for the LCC, Otsu, Otsulog, and Average contours
        self.line_color  = ['red', 'bisque', 'lightsteelblue', 'orange']

        self.font_size   = 15 # label font size
        self.line_width  = 1  # contour line width

        # List of user modifiable plot attributes in this class
        self.user_modifiable_plot_attributes = [
            'Nlevels',
            'cmapname',
            'xy_units', 
            'cbar_units', 
            'xlabel',
            'ylabel',
            'clabel', 
            'line_style',
            'line_color',
            'font_size',
            'line_width'
        ]

        # Scaling options for the data
        self.i_linear        = 0
        self.i_logarithmic   = 1
        self.scaling_options = [None] * 2
        self.scaling_options[self.i_linear]      = 'Linear'
        self.scaling_options[self.i_logarithmic] = 'Logarithmic'
        
        # Set axis and cbar labels
        self.xlabel_plot = None
        self.ylabel_plot = None
        self.clabel_lin  = None
        self.clabel_log  = None

        # Handles for the figure and axes
        self.hF = None
        self.hA = None

    def make_plot_labels(self):
        """
        Generate the labels for the plot axes and colorbar based on current settings.
        """
        self.xlabel_plot = f"{self.xlabel} [{self.xy_units}]"
        self.ylabel_plot = f"{self.ylabel} [{self.xy_units}]"
        self.clabel_lin  = f"{self.clabel} [{self.cbar_units}]"
        self.clabel_log  = f"log$_{{10}}$ {self.clabel} [{self.cbar_units}]"

    def initialise_figure(self, CL):
        """
        Initialize a matplotlib figure with the raw image data and create a colorbar.
        
        Parameters:
        CL (object): An object that contains global settings for all Contour Levels (CL)
        """
        # Setup the figure and axes
        plt.close('all')
        self.hF, self.hA = plt.subplots()

        # Make the plot lables
        self.make_plot_labels()
        self.hA.set_xlabel(self.xlabel_plot, fontsize=self.font_size)
        self.hA.set_ylabel(self.ylabel_plot, fontsize=self.font_size)

        # Generate all the contour levels for the Raw image
        CL.global_levels_raw = np.linspace(self.image_min, self.image_max, self.Nlevels)

        # Set the default background to the Raw image and setup the colorbar
        self.background_cmap = plt.imshow(self.image_raw, cmap=self.cmapname, origin='lower', extent=self.extent)
        self.cbar = plt.colorbar(self.background_cmap, ax=self.hA)
        self.cbar.set_label(self.clabel_lin, fontsize=self.font_size)

    def fix_endianness(self, image):
        """
        Fix endianness issues in the image data if necessary.
        
        Parameters:
        image (numpy.ndarray): The input image data.
        
        Returns:
        numpy.ndarray: The image data with corrected endianness.
        """
        # Ensure the image is a NumPy array
        image = np.asarray(image)

        # Check if the image byte order is not native to the local machine
        if image.dtype.byteorder not in ('=', '|'):
            # Change byte order to native
            image = image.byteswap().newbyteorder()

        return image

    def swap_scaling(self, scaling):
        """
        Swap the scaling method between linear and logarithmic.
        
        Parameters:
        scaling (str): Options for how to scale the data ('Linear' or 'Logarithmic').
        
        Returns:
        tuple: A tuple containing the scale function and the corresponding colorbar label.
        """
        if scaling == self.scaling_options[self.i_logarithmic]:
            scale_function = np.log10
            clabel = self.clabel_log
        else:
            scale_function = lambda x: x
            clabel = self.clabel_lin
            
        return scale_function, clabel

    def set_background_cmap(self, CL, M, selected_data=None, selected_scaling=None, cmapname=None, cmin=None, cmax=None):
        """
        Set the background colormap for the image based on selected data and scaling.
        
        Parameters:
        CL (object): An object that contains global settings for all Contour Levels (CL)
        M (object): An object that contains global settings for a binary mask that masks the data.
        selected_data (str, optional): The selected data to use (raw, smoothed, or no background).
        selected_scaling (str, optional): The selected scaling method ('Linear' or 'Logarithmic').
        cmapname (str, optional): The name of the selected colormap.
        cmin (float, optional): The minimum value for the colorbar.
        cmax (float, optional): The maximum value for the colorbar.
        """
        from matplotlib.colors import Normalize

        # Set defaults for the various inputs
        if selected_data is None:
            selected_data = CL.dataList[CL.i_raw]
        if selected_scaling is None:
            selected_scaling = self.ID.scaling_options[self.ID.i_linear]
        if cmapname is None:
            cmapname = self.cmapname
        if cmin is None:
            cmin = self.cbar_min
        if cmax is None:
            cmax = self.cbar_max

        # The three available background images from which the contours are calculated
        self.image_raw          = np.nan_to_num(self.image_raw, nan=0, posinf=0, neginf=0)
        self.image_smoothed     = np.nan_to_num(self.image_smoothed, nan=0, posinf=0, neginf=0)
        self.image_nobackground = np.nan_to_num(self.image_nobackground, nan=0, posinf=0, neginf=0)

        # Selection of the data
        if selected_data == CL.dataList[CL.i_raw]: # Raw image
            self.image_raw[self.image_raw == 0] = 1e-300
            data = self.image_raw
        elif selected_data == CL.dataList[CL.i_smoothed]: # Smoothed image
            self.image_smoothed[self.image_smoothed == 0] = 1e-300
            data = self.image_smoothed
        elif selected_data == CL.dataList[CL.i_nobackground]: # No Background
            self.image_nobackground[self.image_nobackground == 0] = 1e-300
            data = self.image_nobackground

        # Mask the data if applicable
        data = np.ma.masked_where(M.mask == 0, data)

        # Select the scaling for the data
        scale_function, clabel = self.swap_scaling(selected_scaling)

        # Apply the scaling for the background
        norm = Normalize(vmin=scale_function(np.min(data)), vmax=scale_function(np.max(data)))
        self.background_cmap.set_array(scale_function(data))
        self.background_cmap.set_norm(norm)
    
        # Set the colormap
        self.background_cmap.set_cmap(cmapname)

        # Update colorbar limits
        try:
            self.cbar.mappable.set_clim(scale_function(cmin), scale_function(cmax))
        except ValueError:
            pass
    
        # Redraw the colorbar
        self.cbar.update_normal(self.background_cmap)
        self.cbar.set_label(clabel, fontsize=self.font_size)
#-------------------End ImageData


#-------------------
# Smoothing Class
#-------------------
class Smoothing:
    """
    A collection of smoothing functions and parameters that can be used to smooth
    the image, particularly useful for noisy data.
    """
    
    def __init__(self, ID):
        """
        Initialize the Smoothing class with various smoothing methods and parameters.

        Parameters:
            ID (object): An instance of the ImageData (ID) class.
        """
        self.ID = ID

        # Smoothing methods
        self.i_gaussian  = 0
        self.i_bivspline = 1
        self.i_loess     = 2
        self.i_savgol    = 3
        self.i_movingavg = 4
        self.i_wiener    = 5
        self.i_bilateral = 6
        self.i_totvar    = 7
        self.i_anisodiff = 8
        self.i_nonlocal  = 9
        self.smoothing_methods              = [None] * 10
        self.smoothing_methods[self.i_gaussian]  = 'Gaussian'
        self.smoothing_methods[self.i_bivspline] = 'Bivariate Spline'
        self.smoothing_methods[self.i_loess]     = 'LOESS 2D'
        self.smoothing_methods[self.i_savgol]    = 'Savitzky-Golay'
        self.smoothing_methods[self.i_movingavg] = 'Moving Average'
        self.smoothing_methods[self.i_wiener]    = 'Wiener'
        self.smoothing_methods[self.i_bilateral] = 'Bilateral'
        self.smoothing_methods[self.i_totvar]    = 'Total Variation'
        self.smoothing_methods[self.i_anisodiff] = 'Anisotropic Diffusion'
        self.smoothing_methods[self.i_nonlocal]  = 'Non-Local Means'
        
        # Initial smoothing method
        self.initial_smoothing_method       = self.smoothing_methods[0]

        #---------------------------------------
        # Smoothing parameters for each method
        #---------------------------------------
        self.i_gaussian_sigma               = 0
        #---------------------------------------
        self.i_spline_smoothing_param       = 0
        #---------------------------------------
        self.i_loess_poly_order             = 0
        self.i_loess_locality_frac          = 1
        #---------------------------------------
        self.i_savgol_window_length         = 0
        self.i_savgol_poly_order            = 1
        self.i_savgol_boundary_mode         = 2
        #---------------------------------------
        self.i_moving_avg_kernel_radius     = 0
        #---------------------------------------
        self.i_wiener_mysize                = 0
        self.i_wiener_noise                 = 1
        #---------------------------------------
        self.i_bilateral_sigma_spatial      = 0
        self.i_bilateral_sigma_range        = 1
        #---------------------------------------
        self.i_total_variation_weight       = 0
        self.i_total_variation_n_iter       = 1
        #---------------------------------------
        self.i_anisotropic_diffusion_n_iter = 0
        self.i_anisotropic_diffusion_kappa  = 1
        self.i_anisotropic_diffusion_gamma  = 2
        #---------------------------------------
        self.i_non_local_means_patch_size   = 0
        self.i_non_local_means_h            = 1
        #---------------------------------------

        # Dictionary for the smoothing parameters, their initial/current values, and
        # various other information needed for the interactive smoothing widgets
        self.smoothing_parameters = {
            self.smoothing_methods[self.i_gaussian]: {
                self.i_gaussian_sigma: {
                    'label':     'Sigma',
                    'var_name':  'sigma',
                    'value':     2.0,
                    'min_value': self.ID.small_val,
                    'max_value': self.ID.large_val,
                    'step':      1
                }
            },
            self.smoothing_methods[self.i_bivspline]: {
                self.i_spline_smoothing_param: {
                    'label':     's',
                    'var_name':  's',
                    'value':     self.ID.rounded_min,
                    'min_value': self.ID.small_val,
                    'max_value': self.ID.large_val,
                    'step':      self.ID.rounded_min_magnitude
                }
            },
            self.smoothing_methods[self.i_loess]: {
                self.i_loess_poly_order: {
                    'label':     'Order',
                    'var_name':  'poly_order',
                    'value':     2,
                    'min_value': 0,
                    'max_value': self.ID.large_int
                },
                self.i_loess_locality_frac: {
                    'label':     'Locality Frac',
                    'var_name':  'locality_frac',
                    'value':     0.01,
                    'min_value': self.ID.small_val,
                    'max_value': 1,
                    'step':      0.01
                }
            },
            self.smoothing_methods[self.i_savgol]: {
                self.i_savgol_window_length: {
                    'label':     'Length',
                    'var_name':  'window_length',
                    'value':     10,
                    'min_value': 3, # Needs to be 1 larger than i_savgol_poly_order
                    'max_value': self.ID.large_int
                },
                self.i_savgol_poly_order: {
                    'label':     'Order',
                    'var_name':  'poly_order',
                    'value':     2,
                    'min_value': 0,
                    'max_value': self.ID.large_int
                },
                self.i_savgol_boundary_mode: {
                    'label':    'Boundary',
                    'var_name': 'boundary_mode',
                    'options':  ['nearest', 'constant', 'mirror', 'wrap', 'interp'],
                    'value':    'nearest'
                }
            },
            self.smoothing_methods[self.i_movingavg]: {
                self.i_moving_avg_kernel_radius: {
                    'label':     'Radius',
                    'var_name':  'kernel_radius',
                    'value':     3.0,
                    'min_value': 1,
                    'max_value': self.ID.large_int
                }
            },
            self.smoothing_methods[self.i_wiener]: {
                self.i_wiener_mysize: {
                    'label':     'Size',
                    'var_name':  'mysize',
                    'value':     5,
                    'min_value': 1,
                    'max_value': self.ID.large_int
                },
                self.i_wiener_noise: {
                    'label':     'Noise',
                    'var_name':  'noise',
                    'value':     0.05,
                    'min_value': self.ID.small_val,
                    'max_value': self.ID.large_val,
                    'step':      0.1
                }
            },
            self.smoothing_methods[self.i_bilateral]: {
                self.i_bilateral_sigma_spatial: {
                    'label':     'Sigma_x',
                    'var_name':  'sigma_spatial',
                    'value':     1.0,
                    'min_value': self.ID.small_val,
                    'max_value': self.ID.large_val,
                    'step':      0.1
                },
                self.i_bilateral_sigma_range: {
                    'label':     'Sigma_I',
                    'var_name':  'sigma_range',
                    'value':     1.0,
                    'min_value': self.ID.small_val,
                    'max_value': self.ID.large_val,
                    'step':      0.1
                }
            },
            self.smoothing_methods[self.i_totvar]: {
                self.i_total_variation_weight: {
                    'label':     'Weight',
                    'var_name':  'weight',
                    'value':     0.5,
                    'min_value': 0.0,
                    'max_value': 1.0,
                    'step':      0.1
                },
                self.i_total_variation_n_iter: {
                    'label':     'Iterations',
                    'var_name':  'n_iter',
                    'value':     100,
                    'min_value': 1,
                    'max_value': self.ID.large_int
                }
            },
            self.smoothing_methods[self.i_anisodiff]: {
                self.i_anisotropic_diffusion_n_iter: {
                    'label':     'Iterations',
                    'var_name':  'n_iter',
                    'value':     10,
                    'min_value': 1,
                    'max_value': self.ID.large_int
                },
                self.i_anisotropic_diffusion_kappa: {
                    'label':     'Kappa',
                    'var_name':  'kappa',
                    'value':     50.0,
                    'min_value': self.ID.small_val,
                    'max_value': self.ID.large_val,
                    'step':      0.1
                },
                self.i_anisotropic_diffusion_gamma: {
                    'label':     'Gamma',
                    'var_name':  'gamma',
                    'value':     0.1,
                    'min_value': self.ID.small_val,
                    'max_value': self.ID.large_val,
                    'step':      0.01
                }
            },
            self.smoothing_methods[self.i_nonlocal]: {
                self.i_non_local_means_patch_size: {
                    'label':     'Size',
                    'var_name':  'patch_size',
                    'value':     10,
                    'min_value': 1,
                    'max_value': self.ID.large_int
                },
                self.i_non_local_means_h: {
                    'label':     'h',
                    'var_name':  'h',
                    'value':     self.ID.rounded_min,
                    'min_value': self.ID.small_val,
                    'max_value': self.ID.large_val,
                    'step':      self.ID.rounded_min_magnitude
                }
            }
        }

    def smooth_data(self, X=None, Y=None, image=None, method=None):
        """
        Smooth the input data using different smoothing methods.
    
        Parameters:
        - X, Y: 2D arrays representing the grid coordinates
        - image: 2D array corresponding to the values at (X, Y)
        - method: String specifying the smoothing method
    
        Returns:
        - image_smoothed: 2D array of smoothed values
        """
        if X is None:
            X = self.ID.X
        if Y is None:
            Y = self.ID.Y
        if image is None:
            image = self.ID.image_raw
        if method is None:
            method = self.initial_smoothing_method
    
        if method == self.smoothing_methods[self.i_gaussian]:
            # Gaussian smoothing
            sigma = self.smoothing_parameters[method][self.i_gaussian_sigma]['value']
            from scipy.ndimage import gaussian_filter
            image_smoothed = gaussian_filter(image, sigma=sigma)
    
        elif method == self.smoothing_methods[self.i_bivspline]:
            # Bivariate spline smoothing
            smoothing_param = self.smoothing_parameters[method][self.i_spline_smoothing_param]['value']
            from scipy.interpolate import RectBivariateSpline
            image_smoothed = RectBivariateSpline(X[0, :], Y[:, 0], image, s=smoothing_param)(X[0, :], Y[:, 0])
    
        elif method == self.smoothing_methods[self.i_loess]:
            # Local regression (LOESS) smoothing
            image_flat = image.flatten()
            poly_order = self.smoothing_parameters[method][self.i_loess_poly_order]['value']
            locality_frac = self.smoothing_parameters[method][self.i_loess_locality_frac]['value']
            from loess.loess_2d import loess_2d
            image2d_smoothed, _ = loess_2d(X.flatten(), Y.flatten(), image_flat, degree=poly_order, frac=locality_frac)
            image_smoothed = image2d_smoothed.reshape(image.shape)
    
        elif method == self.smoothing_methods[self.i_savgol]:
            # Savitzky-Golay smoothing
            window_length = self.smoothing_parameters[method][self.i_savgol_window_length]['value']
            poly_order = self.smoothing_parameters[method][self.i_savgol_poly_order]['value']
            boundary_mode = self.smoothing_parameters[method][self.i_savgol_boundary_mode]['value']
            from scipy.signal import savgol_filter
            image_smoothed = savgol_filter(
                savgol_filter(image, window_length=window_length, polyorder=poly_order, axis=0, mode=boundary_mode),
                window_length=window_length, polyorder=poly_order, axis=1, mode=boundary_mode)
    
        elif method == self.smoothing_methods[self.i_movingavg]:
            # Moving average smoothing
            kernel_radius = self.smoothing_parameters[method][self.i_moving_avg_kernel_radius]['value']
    
            def circular_kernel(radius):
                x = np.arange(-radius, radius + 1)
                y = np.arange(-radius, radius + 1)
                x, y = np.meshgrid(x, y)
                circular_mask = x**2 + y**2 <= radius**2
                kernel = np.zeros_like(x, dtype=float)
                kernel[circular_mask] = 1.0
                return kernel / np.sum(kernel)
    
            from scipy.signal import convolve2d
            image_smoothed = convolve2d(image, circular_kernel(kernel_radius), mode='same', boundary='symm')
    
        elif method == self.smoothing_methods[self.i_wiener]:
            # Wiener filtering
            mysize = self.smoothing_parameters[method][self.i_wiener_mysize]['value']
            noise = self.smoothing_parameters[method][self.i_wiener_noise]['value']
            from scipy.signal import wiener
            image_smoothed = wiener(image, mysize=(mysize, mysize), noise=noise)
    
        elif method == self.smoothing_methods[self.i_bilateral]:
            # Bilateral filtering
            sigma_spatial = self.smoothing_parameters[method][self.i_bilateral_sigma_spatial]['value']
            sigma_range = self.smoothing_parameters[method][self.i_bilateral_sigma_range]['value']
            from skimage.restoration import denoise_bilateral
            image_smoothed = denoise_bilateral(image, sigma_color=sigma_range, sigma_spatial=sigma_spatial)
    
        elif method == self.smoothing_methods[self.i_totvar]:
            # Total variation denoising
            alpha = self.smoothing_parameters[method][self.i_total_variation_weight]['value']
            n_iter = self.smoothing_parameters[method][self.i_total_variation_n_iter]['value']
            from skimage.restoration import denoise_tv_chambolle
            image_smoothed = denoise_tv_chambolle(image, weight=alpha, max_num_iter=n_iter)
    
        elif method == self.smoothing_methods[self.i_anisodiff]:
            # Anisotropic diffusion
            n_iter = self.smoothing_parameters[method][self.i_anisotropic_diffusion_n_iter]['value']
            kappa = self.smoothing_parameters[method][self.i_anisotropic_diffusion_kappa]['value']
            gamma = self.smoothing_parameters[method][self.i_anisotropic_diffusion_gamma]['value']
            from skimage.restoration import denoise_tv_bregman
            image_smoothed = denoise_tv_bregman(image, weight=gamma, max_num_iter=n_iter, eps=kappa)
    
        elif method == self.smoothing_methods[self.i_nonlocal]:
            # Non-local means denoising
            patch_size = self.smoothing_parameters[method][self.i_non_local_means_patch_size]['value']
            h = self.smoothing_parameters[method][self.i_non_local_means_h]['value']
            from skimage.restoration import denoise_nl_means
            image_smoothed = denoise_nl_means(image, patch_size=patch_size, h=h)
    
        else:
            # No smoothing
            print('No matching methods')
            image_smoothed = image
    
        image_smoothed = np.nan_to_num(image_smoothed, nan=0, posinf=0, neginf=0)
        image_smoothed = np.clip(image_smoothed, 1e-300, None)
        
        return image_smoothed
                    
    def initialise_image_smoothed(self):
        """
        Initialize the smoothed image using the initial smoothing method.
        """
        self.ID.image_smoothed = self.smooth_data(method=self.initial_smoothing_method)
#-------------------End Smoothing


#-------------------
# Remove Background
#-------------------
class NoBackground:
    """
    A class to remove background from an image using specified filter and padding parameters.
    """
    
    def __init__(self, ID):
        """
        Initialize the NoBackground class with default filter and padding parameters.

        Parameters:
            ID (object): An instance of the ImageData (ID) class.
        """
        self.ID = ID
        
        # Define the list of filter shapes
        self.i_disk_filter   = 0
        self.i_square_filter = 1
        self.filter_shape_options                  = [None] * 2
        self.filter_shape_options[self.i_disk_filter]   = 'disk'
        self.filter_shape_options[self.i_square_filter] = 'square'
        
        # Define the list of padding modes for the image edges
        self.i_reflect   = 0
        self.i_symmetric = 1
        self.i_constant  = 2
        self.padding_options                   = [None] * 3
        self.padding_options[self.i_reflect]   = 'reflect'
        self.padding_options[self.i_symmetric] = 'symmetric'
        self.padding_options[self.i_constant]  = 'constant'
        
        # Set the initial/default values
        self.initial_filter_size_max  = np.max(self.ID.image_raw.shape)
        self.initial_filter_size      = int(self.initial_filter_size_max/6)
        self.initial_filter_size_step = int(self.initial_filter_size_max/100)
        self.initial_filter_shape     = self.filter_shape_options[0]
        self.initial_pad_mode         = self.padding_options[0]
        self.initial_pad_value        = self.ID.image_min
  
        # Dictionary for the nobackground parameters and relevant information for the widgets
        self.i_filter_shape = 0
        self.i_filter_size  = 1
        self.i_pad_mode     = 2
        self.i_pad_value    = 3
        self.nobackground_parameters = {
            self.i_filter_shape: {
                'description': 'Filter Shape',
                'var_name':    'filter_shape',
                'value':       self.initial_filter_shape,
                'options':     self.filter_shape_options,
                'layout':      {'width': '200px'}
            },
            self.i_filter_size: {
                'description': 'Size',
                'var_name':    'filter_size',
                'value':       self.initial_filter_size,
                'min_value':   1,
                'max_value':   self.initial_filter_size_max,
                'step':        self.initial_filter_size_step
            },
            self.i_pad_mode: {
                'description': 'Padding',
                'var_name':    'pad_mode',
                'value':       self.initial_pad_mode,
                'options':     self.padding_options,
                'layout':      {'width': '200px'}
            },
            self.i_pad_value: {
                'description': 'Pad Value',
                'var_name':    'pad_value',
                'value':       self.initial_pad_value,
                'min_value':   self.ID.small_val,
                'max_value':   self.ID.large_val,
                'step':        1
            }
        }
    
    def remove_background(self, image, filter_size=None, filter_shape=None, pad_mode=None, pad_value=None):
        """
        Remove the background from an image using the specified filter and padding parameters.

        Parameters:
            image (2D array): The input image from which to remove the background.
            filter_size (int, optional): The size of the filter used for background estimation. Defaults to self.initial_filter_size.
            filter_shape (str, optional): The shape of the filter used for background estimation. Defaults to self.initial_filter_shape.
            pad_mode (str, optional): The padding mode used at the image boundaries. Defaults to self.initial_pad_mode.
            pad_value (float, optional): The value used for constant padding. Defaults to self.initial_pad_value.

        Returns:
            tuple: A tuple containing:
                - image_nobackground (2D array): The image with the background removed.
                - background (2D array): The estimated background.
                - image_difference (2D array): The difference between the original image and the background-removed image.
        """
        from skimage.filters import threshold_otsu, rank
        from skimage import img_as_ubyte
        from skimage.morphology import disk, square

        # Set default values for the inputs
        if filter_size is None:
            filter_size = self.initial_filter_size
        if filter_shape is None:
            filter_shape = self.initial_filter_shape
        if pad_mode is None:
            pad_mode = self.initial_pad_mode
        if pad_value is None:
            pad_value = self.initial_pad_value

        # Normalise image to [0, 1]
        image_normalized = (image - self.ID.image_min) / (self.ID.image_max - self.ID.image_min)
    
        # Convert to grayscale image
        grayscale_image = np.abs(image_normalized)
    
        # Ensure the image is in the range [-1, 1]
        grayscale_image = np.clip(grayscale_image, -1, 1)
    
        # Convert image to uint8
        image_uint8 = img_as_ubyte(grayscale_image)
    
        # Apply thresholding to identify background
        threshold_value = threshold_otsu(image_uint8)
        binary_background = image_uint8 > threshold_value
    
        # Apply padding to boundaries to reduce filter artefacts
        if pad_mode == self.padding_options[self.i_reflect]:
            padded_image = np.pad(grayscale_image, filter_size, mode=self.padding_options[self.i_reflect])
        elif pad_mode == self.padding_options[self.i_symmetric]:
            padded_image = np.pad(grayscale_image, filter_size, mode=self.padding_options[self.i_symmetric])
        elif pad_mode == self.padding_options[self.i_constant]:
            padded_image = np.pad(grayscale_image, filter_size, constant_values=pad_value)
        else:
            raise ValueError(f"Invalid padding mode. Choose one of the following: {', '.join(self.padding_options.values())}.")
    
        # Crop the padded image to match the raw size
        cropped_image = padded_image[:grayscale_image.shape[0], :grayscale_image.shape[1]]
    
        # Convert cropped image to uint8
        cropped_image_uint8 = img_as_ubyte(cropped_image)
    
        # Choose the filter shape based on the specified option
        if filter_shape == self.filter_shape_options[self.i_disk_filter]:
            filter_shape_obj = disk(filter_size)
        elif filter_shape == self.filter_shape_options[self.i_square_filter]:
            filter_shape_obj = square(filter_size)
        else:
            raise ValueError(f"Invalid filter shape. Choose one of the following: {', '.join(self.filter_shape_options.values())}.")
    
        # Apply rank filter to estimate background
        background = rank.mean(cropped_image_uint8, filter_shape_obj) / 255.0 * (np.max(image) - np.min(image)) + np.min(image)
    
        # Subtract estimated background from the raw image in the raw data space
        image_nobackground = image - background
    
        # Clip the resulting image to ensure it stays within a reasonable range
        image_nobackground = np.clip(image_nobackground, self.ID.image_min, self.ID.image_max)
    
        # Calculate the difference and error
        image_difference = np.abs(image_nobackground - image)
        
        return image_nobackground, background, image_difference

    def initialise_image_nobackground(self):
        """
        Initialize the background-removed image using the initial filter and padding parameters.
        """
        self.ID.image_nobackground, self.ID.estimated_background, self.ID.image_difference = self.remove_background(self.ID.image_raw, self.initial_filter_size, self.initial_filter_shape, self.initial_pad_mode)
#-------------------End NoBackground


#-------------------
# Mask
#-------------------
class Mask:
    """
    A class to create and manage masks for image processing, supporting different mask shapes and edge detection.
    """

    def __init__(self, ID):
        """
        Initialize the Mask class with default mask parameters.

        Parameters:
            ID (object): An instance of the ImageData (ID) class.
        """
        self.ID = ID
        
        # Define the list of filter shapes
        self.i_circle_mask = 0
        self.i_square_mask = 1
        self.mask_shape_options                     = [None] * 2
        self.mask_shape_options[self.i_circle_mask] = 'circle'
        self.mask_shape_options[self.i_square_mask] = 'square'

        # Set default values for the mask
        self.initial_mask_xpos   = (self.ID.xmin + self.ID.xmax) / 2
        self.initial_mask_ypos   = (self.ID.ymin + self.ID.ymax) / 2
        self.initial_mask_shape  = self.mask_shape_options[0]
        self.initial_mask_size   = int(min(self.ID.image_raw.shape) / 4)
        self.initial_x_step_size = (self.ID.xmax - self.ID.xmin) / 100
        self.initial_y_step_size = (self.ID.ymax - self.ID.ymin) / 100

        # Set the default mask to 1s (i.e. no masking of data)
        self.mask = np.ones_like(self.ID.image_raw)

    def create_mask(self, mask_xpos=None, mask_ypos=None, mask_shape=None, mask_size=None):
        """
        Create a mask based on the specified parameters.

        Parameters:
            mask_xpos (float, optional): The x position of the mask center. Defaults to self.initial_mask_xpos.
            mask_ypos (float, optional): The y position of the mask center. Defaults to self.initial_mask_ypos.
            mask_shape (str, optional): The shape of the mask ('circle' or 'square'). Defaults to self.initial_mask_shape.
            mask_size (int, optional): The size (radius for circle, half-side length for square) of the mask. Defaults to self.initial_mask_size.

        Returns:
            2D array: The created mask.
        """
        # Set default values for the inputs
        if mask_xpos is None:
            mask_xpos = self.initial_mask_xpos
        if mask_ypos is None:
            mask_ypos = self.initial_mask_ypos
        if mask_shape is None:
            mask_shape = self.initial_mask_shape
        if mask_size is None:
            mask_size = self.initial_mask_size

        # Find the shape of the original image
        image_shape = self.ID.image_raw.shape
        
        # Find the nearest indices to the given center point
        ix_center = np.abs(self.ID.x - mask_xpos).argmin()
        iy_center = np.abs(self.ID.y - mask_ypos).argmin()
    
        # Create meshgrid of indices (row major order)
        ix, iy = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
    
        # Calculate distances from center
        distances = np.sqrt((ix - ix_center) ** 2 + (iy - iy_center) ** 2)
    
        # Create the mask based on the selected shape
        if mask_shape == self.mask_shape_options[self.i_circle_mask]:
            mask = np.where(distances <= mask_size, 1, 0)
        elif mask_shape == self.mask_shape_options[self.i_square_mask]:
            mask    = np.zeros(image_shape)
            start_x = max(0, ix_center - mask_size)
            end_x   = min(image_shape[1], ix_center + mask_size)
            start_y = max(0, iy_center - mask_size)
            end_y   = min(image_shape[0], iy_center + mask_size)
            mask[start_y:end_y, start_x:end_x] = 1
        else:
            raise ValueError(f"Invalid mask shape. Choose one of the following: {', '.join(self.mask_shape_options)}")
    
        return mask

    def find_edges(self, mask, thickness=2):
        """
        Find the edges of the given mask with a specified thickness.

        Parameters:
            mask (2D array): The input mask array.
            thickness (int, optional): The thickness of the edges. Defaults to 2.

        Returns:
            2D array: The edges of the mask.
        """
        from scipy.ndimage import binary_erosion, binary_dilation
        
        # Ensure the mask is a boolean array
        mask = mask.astype(bool)
        
        # Dilate the mask to create a region outside of the original mask
        dilated_mask = binary_dilation(mask, iterations=thickness)
        
        # Compute the edges by subtracting the original mask from the dilated mask
        custom_edges = dilated_mask & ~mask
        
        # Compute the border of the dilated mask
        dilated_border = dilated_mask & ~binary_erosion(dilated_mask)
        
        # Create a mask representing the area where the dilated mask extends beyond the array boundary
        boundary_mask        = np.zeros_like(mask, dtype=bool)
        boundary_mask[0, :]  = True
        boundary_mask[-1, :] = True
        boundary_mask[:, 0]  = True
        boundary_mask[:, -1] = True
        
        # Combine the edges with the boundary where the dilated mask extends beyond the array borders
        edges = custom_edges | (dilated_border & boundary_mask)
        
        return edges
#-------------------End Mask


#-------------------
# Contour Levels
#-------------------
class ContourLevels:
    """
    A class for managing all contour levels and generating contour plots for different background images.
    """
    
    def __init__(self, ID, M):
        """
        Initialize the ContourLevels class with default parameters.

        Parameters:
            ID (object): An instance of the ImageData (ID) class.
            M  (object): An instance of the Mask (M) class.
        """
        self.ID = ID
        self.M  = M

        # Labels for the 3 different background images (data types)
        # (rows in the checkbox grid)
        self.i_raw          = 0
        self.i_smoothed     = 1
        self.i_nobackground = 2
        self.dataList                      = [None] * 3
        self.dataList[self.i_raw]          = 'Raw'
        self.dataList[self.i_smoothed]     = 'Smoothed'
        self.dataList[self.i_nobackground] = 'No Background'
        
        # Labels for the 4 different contour calculations (level types)
        # (columns in the checkbox grid)
        self.i_lcc      = 0
        self.i_otsu     = 1
        self.i_otsulog  = 2
        self.i_average  = 3
        self.levelList                 = [None] * 4
        self.levelList[self.i_lcc]     = 'LCC'
        self.levelList[self.i_otsu]    = 'Otsu'
        self.levelList[self.i_otsulog] = 'OtsuLog'
        self.levelList[self.i_average] = 'Average'
        
        # Define row and column labels for checkboxes
        self.row_labels    = [label + ':' for label in self.dataList]
        self.column_labels = [''] + self.levelList

        # Make containers for the global contour levels for each background image
        self.global_levels_raw          = np.linspace(self.ID.image_min, self.ID.image_max, self.ID.Nlevels)
        self.global_levels_smoothed     = None
        self.global_levels_nobackground = None

        # Save the spacing between levels
        self.delta_levels               = [None] * len(self.dataList)
        self.delta_levels[self.i_raw]   = self.global_levels_raw[1] - self.global_levels_raw[0]

        # Set the initial threshold value for finding the average contour
        self.initial_averaging_threshold_percentage = 20

        # Contour objects for all of the global contour levels
        self.CS              = None
        self.CS_smoothed     = None
        self.CS_nobackground = None

        # Empty dictionaries to be filled with contour objects and properties
        self.contour_colors  = None
        self.contour_styles  = None
        self.contour_lines   = None
        self.contour_levels  = None

        # Containers to hold the current value of the calculated contour levels
        self.lcc_level                   = None
        self.lcc_level_smoothed          = None
        self.lcc_level_nobackground      = None
        self.otsu_level                  = None
        self.otsu_level_smoothed         = None
        self.otsu_level_nobackground     = None
        self.otsu_log_level              = None
        self.otsu_log_level_smoothed     = None
        self.otsu_log_level_nobackground = None
        self.average_level               = None
        self.average_level_smoothed      = None
        self.average_level_nobackground  = None

        # Containers to hold the contour objects of the calculated contour levels
        self.lcc_contour                   = None
        self.otsu_contour                  = None
        self.otsu_log_contour              = None
        self.average_contour               = None
        self.lcc_contour_smoothed          = None
        self.otsu_contour_smoothed         = None
        self.otsu_log_contour_smoothed     = None
        self.average_contour_smoothed      = None
        self.lcc_contour_nobackground      = None
        self.otsu_contour_nobackground     = None
        self.otsu_log_contour_nobackground = None
        self.average_contour_nobackground  = None

    def setup_contour_dictionaries(self):
        """
        Sets up dictionaries to map contour names to line colors, styles, and calculated contour objects and levels.
        """
        # Define a dictionary to map contour names to line colors
        self.contour_colors = {
            self.levelList[self.i_lcc]:     {'color': self.ID.line_color[self.i_lcc]},
            self.levelList[self.i_otsu]:    {'color': self.ID.line_color[self.i_otsu]},
            self.levelList[self.i_otsulog]: {'color': self.ID.line_color[self.i_otsulog]},
            self.levelList[self.i_average]: {'color': self.ID.line_color[self.i_average]}
        }
        
        # Define a dictionary to map contour names to line styles
        self.contour_styles = {
            self.dataList[self.i_raw]:          {'linestyle': self.ID.line_style[self.i_raw],          'color': self.ID.line_color[self.i_lcc]},
            self.dataList[self.i_smoothed]:     {'linestyle': self.ID.line_style[self.i_smoothed],     'color': self.ID.line_color[self.i_otsu]},
            self.dataList[self.i_nobackground]: {'linestyle': self.ID.line_style[self.i_nobackground], 'color': self.ID.line_color[self.i_average]}
        }
        
        # Create a dictionary to store the calculated contour objects
        self.contour_lines = {
            self.dataList[self.i_raw]: {
                self.levelList[self.i_lcc]:     self.lcc_contour,
                self.levelList[self.i_otsu]:    self.otsu_contour,
                self.levelList[self.i_otsulog]: self.otsu_log_contour,
                self.levelList[self.i_average]: self.average_contour
            },
            self.dataList[self.i_smoothed]: {
                self.levelList[self.i_lcc]:     self.lcc_contour_smoothed,
                self.levelList[self.i_otsu]:    self.otsu_contour_smoothed,
                self.levelList[self.i_otsulog]: self.otsu_log_contour_smoothed,
                self.levelList[self.i_average]: self.average_contour_smoothed
            },
            self.dataList[self.i_nobackground]: {
                self.levelList[self.i_lcc]:     self.lcc_contour_nobackground,
                self.levelList[self.i_otsu]:    self.otsu_contour_nobackground,
                self.levelList[self.i_otsulog]: self.otsu_log_contour_nobackground,
                self.levelList[self.i_average]: self.average_contour_nobackground
            }
        }
        
        # Create a dictionary to store the calculated contour levels
        self.contour_levels = {
            self.dataList[self.i_raw]: {
                self.levelList[self.i_lcc]:     self.lcc_level,
                self.levelList[self.i_otsu]:    self.otsu_level,
                self.levelList[self.i_otsulog]: self.otsu_log_level,
                self.levelList[self.i_average]: self.average_level
            },
            self.dataList[self.i_smoothed]: {
                self.levelList[self.i_lcc]:     self.lcc_level_smoothed,
                self.levelList[self.i_otsu]:    self.otsu_level_smoothed,
                self.levelList[self.i_otsulog]: self.otsu_log_level_smoothed,
                self.levelList[self.i_average]: self.average_level_smoothed
            },
            self.dataList[self.i_nobackground]: {
                self.levelList[self.i_lcc]:     self.lcc_level_nobackground,
                self.levelList[self.i_otsu]:    self.otsu_level_nobackground,
                self.levelList[self.i_otsulog]: self.otsu_log_level_nobackground,
                self.levelList[self.i_average]: self.average_level_nobackground
            }
        }
    
    def update_contour_levels(self):
        """
        Updates the stored contour levels for raw, smoothed, and no-background data types based on current calculations.
        """
        # Update the Raw contours
        self.contour_levels[self.dataList[self.i_raw]][self.levelList[self.i_lcc]]     = self.lcc_level,
        self.contour_levels[self.dataList[self.i_raw]][self.levelList[self.i_otsu]]    = self.otsu_level
        self.contour_levels[self.dataList[self.i_raw]][self.levelList[self.i_otsulog]] = self.otsu_log_level
        self.contour_levels[self.dataList[self.i_raw]][self.levelList[self.i_average]] = self.average_level

        # Update the Smoothed contours
        self.contour_levels[self.dataList[self.i_smoothed]][self.levelList[self.i_lcc]]     = self.lcc_level_smoothed,
        self.contour_levels[self.dataList[self.i_smoothed]][self.levelList[self.i_otsu]]    = self.otsu_level_smoothed
        self.contour_levels[self.dataList[self.i_smoothed]][self.levelList[self.i_otsulog]] = self.otsu_log_level_smoothed
        self.contour_levels[self.dataList[self.i_smoothed]][self.levelList[self.i_average]] = self.average_level_smoothed

        # Update the NoBackground contours
        self.contour_levels[self.dataList[self.i_nobackground]][self.levelList[self.i_lcc]]     = self.lcc_level_nobackground,
        self.contour_levels[self.dataList[self.i_nobackground]][self.levelList[self.i_otsu]]    = self.otsu_level_nobackground
        self.contour_levels[self.dataList[self.i_nobackground]][self.levelList[self.i_otsulog]] = self.otsu_log_level_nobackground
        self.contour_levels[self.dataList[self.i_nobackground]][self.levelList[self.i_average]] = self.average_level_nobackground

    def update_global_levels_smoothed(self):
        """
        Updates the global contour levels and level spacing for the smoothed image based on its minimum and maximum values.
        """
        # Find the min and max of the Smoothed image
        self.ID.image_min_smoothed  = np.min(self.ID.image_smoothed)
        self.ID.image_max_smoothed  = np.max(self.ID.image_smoothed)

        # Calculate the global contour levels for the smoothed image and get the level spacing
        self.global_levels_smoothed = np.linspace(self.ID.image_min_smoothed, self.ID.image_max_smoothed, self.ID.Nlevels)
        self.delta_levels[self.i_smoothed]  = self.global_levels_smoothed[1] - self.global_levels_smoothed[0]

    def update_global_levels_nobackground(self):
        """
        Updates the global contour levels and level spacing for the no-background image based on its minimum and maximum values.
        """
        # Find the min and max of the NoBackground image
        self.ID.image_min_nobackground  = np.min(self.ID.image_nobackground)
        self.ID.image_max_nobackground  = np.max(self.ID.image_nobackground)

        # Calculate the global contour levels for the smoothed image and get the level spacing
        self.global_levels_nobackground = np.linspace(self.ID.image_min_nobackground, self.ID.image_max_nobackground, self.ID.Nlevels)
        self.delta_levels[self.i_nobackground]  = self.global_levels_nobackground[1] - self.global_levels_nobackground[0]
        
    def make_global_contour_lines(self):
        """
        Generates global contour lines for raw, smoothed, and no-background images using the current global contour levels.
        """
        # Update or initialise the global contours for the Smoothed and NoBackground images
        self.update_global_levels_smoothed()
        self.update_global_levels_nobackground()

        # Contour objects for all of the global contour levels
        self.CS              = plt.contour(self.ID.X, self.ID.Y, self.ID.image_raw,          levels=self.global_levels_raw,              linewidths=0)
        self.CS_smoothed     = plt.contour(self.ID.X, self.ID.Y, self.ID.image_smoothed,     levels=self.global_levels_smoothed,     linewidths=0)
        self.CS_nobackground = plt.contour(self.ID.X, self.ID.Y, self.ID.image_nobackground, levels=self.global_levels_nobackground, linewidths=0)

    def update_contour_lines(self, mask, update_dataList_set=None, update_levelList_set=None):
        """
        Updates the contour lines based on the provided mask and specified data and level sets. 
        It removes existing contours and generates new ones with updated data and styles.
        """
        # Remove any contour lines that need to be updated
        for key, value in self.contour_lines.items():
            for contour_name, contour in value.items():
                if update_dataList_set is None or key in update_dataList_set:
                    if update_levelList_set is None or contour_name in update_levelList_set:
                        if contour:  # Check if the contour exists
                            try:
                                contour.remove()
                            except ValueError:
                                pass  # Ignore the error if contour is not found
    
        # Update contours with new data
        for key, value in self.contour_lines.items():
            for contour_name, contour in value.items():
                if update_dataList_set is None or key in update_dataList_set:
                    if update_levelList_set is None or contour_name in update_levelList_set:

                        # Get levels from contour_levels dictionary
                        levels = [self.contour_levels[key][contour_name]]
                        if key == self.dataList[self.i_raw]:
                            image_masked = np.ma.masked_where(mask == 0, self.ID.image_raw)
                        elif key == self.dataList[self.i_smoothed]:
                            image_masked = np.ma.masked_where(mask == 0, self.ID.image_smoothed)
                        elif key == self.dataList[self.i_nobackground]:
                            image_masked = np.ma.masked_where(mask == 0, self.ID.image_nobackground)
                        
                        # Get linestyle and color from contour_styles dictionary based on contour name
                        linestyle = self.contour_styles[key]['linestyle']
                        color = self.contour_colors[contour_name]['color']

                        # Generate the contour
                        contours = plt.contour(self.ID.X, self.ID.Y, image_masked, levels=levels, linewidths=self.ID.line_width, linestyles=linestyle, colors=color)
                        self.contour_lines[key][contour_name] = contours

    #-------------------
    # Average
    #-------------------
    def find_average_level(self, image, threshold_percentage=None):
        """
        Finds the average contour level in the given image based on the specified threshold percentage. 
        It calculates the gradient magnitude, thresholds it, labels connected components, 
        and finds contours to compute the average level.
        """
        from scipy.ndimage import gaussian_gradient_magnitude, label, find_objects
        from skimage.measure import find_contours

        # Set default threshold value
        if threshold_percentage is None:
            threshold_percentage = self.initial_averaging_threshold_percentage
        
        # Calculate the gradient magnitude
        gradient_magnitude = gaussian_gradient_magnitude(image, sigma=1.0)
    
        # Calculate the actual threshold value based on the percentage
        threshold_value = np.percentile(gradient_magnitude[gradient_magnitude != 0], threshold_percentage)
    
        # Threshold the gradient magnitude
        significant_gradients = gradient_magnitude < threshold_value
    
        # Label connected components in significant_gradients
        labeled_gradients, num_features = label(significant_gradients)
    
        # Find bounding boxes for each labeled region
        bounding_boxes = find_objects(labeled_gradients)
    
        # Assuming the largest region corresponds to the signal, exclude it
        if num_features > 1:
            largest_region_index = np.argmax([np.prod(box[0].stop - box[0].start) for box in bounding_boxes])
            significant_gradients[labeled_gradients == largest_region_index + 1] = False
    
        # Find contours in the raw data within significant_gradients
        # The contour level 0.5 represents the boundary between True (1) and False (0) in binary images
        contours = find_contours(significant_gradients, 0.5)
    
        # Calculate the average contour level
        average_level = np.mean([np.mean(image[np.round(contour[:, 0]).astype(int), np.round(contour[:, 1]).astype(int)]) for contour in contours])
    
        return average_level

    #-------------------
    # Find Closed Contours
    #-------------------
    def is_contour_closed(self, X, Y, image, CS, global_levels, mask=None):
        """
        Determines if contours at specified global levels are closed or open, 
        taking into account the presence of a mask and the edges of the image. 
        Returns the level of the last closed contour.
        """
        # Find the edges of the mask if present
        if mask is not None:
            edges = self.M.find_edges(mask)
        else:
            edges = None

        # Array to hold boolean value of whether a contour is closed
        closed = [None] * len(np.atleast_1d(global_levels))  # Ensure global_levels is at least 1-dimensional
    
        for i, (level, path) in enumerate(zip(CS.levels, CS.get_paths())):
            # Disregard contours with less than 3 unique vertices
            if len(np.unique(path.vertices, axis=0)) < 3 and len(path.vertices) > 0:
                closed[i] = False
            else:
                # Get the contour vertices
                cts = path.vertices
                
                if mask is not None:
                    # Map contour vertices onto the nearest grid points
                    ix_cts = ((cts[:, 0] - self.ID.xmin) / self.ID.dx).astype(int)
                    iy_cts = ((cts[:, 1] - self.ID.ymin) / self.ID.dy).astype(int)
                    
                    # Round indices to the nearest grid points
                    ix_cts = np.clip(ix_cts, 0, len(self.ID.x) - 1)
                    iy_cts = np.clip(iy_cts, 0, len(self.ID.y) - 1)

                if mask is None or np.any(mask[iy_cts, ix_cts] == 1): 
                    if mask is not None and np.any(edges[iy_cts, ix_cts] == 1):
                        # If the contour intersects the mask, consider it open
                        closed[i] = False
                    elif mask is None and ((self.ID.xmin in cts[:, 0]) or (self.ID.xmax in cts[:, 0]) or (self.ID.ymin in cts[:, 1]) or (self.ID.ymax in cts[:, 1])):
                        # If the contour intersects the bounding box, consider it open
                        closed[i] = False
                    else:
                        # If not, consider it closed
                        closed[i] = True
    
        # Find the index of the first True element after the last False element
        # (i.e. find the Last Closed Contour, LCC)
        if False in closed:
            index = len(closed) - closed[::-1].index(False)
        else:
            index = len(closed) - 1  # Take the last element
    
        return global_levels[index]

    #-------------------
    # OTSU theshold
    #-------------------
    def threshold_otsu_positive(self, image, scaling=None):
        """
        Computes the Otsu threshold for positive values in the given image. 
        Optionally applies logarithmic scaling before thresholding.
        """
        from skimage.filters import threshold_otsu

        if scaling is None:
            scaling = self.ID.scaling_options[self.ID.i_linear]
            
        data = image[image > 0]
        
        if scaling==self.ID.scaling_options[self.ID.i_logarithmic]:
            data = np.log10(image[image > 0])
            
        return threshold_otsu(data)
    #-------------------

    #-------------------
    # Calculate Raw Levels
    #-------------------    
    def calculate_lcc_level(self):
        """
        Calculates the Last Closed Contour (LCC) level for the raw image data. 
        Generates contour lines and determines if they are closed, updating the LCC level.
        """
        self.CS = plt.contour(self.ID.X, self.ID.Y, self.ID.image_raw, levels=self.global_levels_raw, linewidths=0)
        self.lcc_level = self.is_contour_closed(self.ID.X, self.ID.Y, self.ID.image_raw, self.CS, self.global_levels_raw, self.M.mask)
    
    def calculate_otsu_level(self):
        """
        Calculates the Otsu threshold level for the raw image data and updates the Otsu level.
        """
        self.otsu_level = self.threshold_otsu_positive(self.ID.image_raw)
    
    def calculate_otsu_log_level(self):
        """
        Calculates the Otsu threshold level for the logarithmically scaled raw image data 
        and updates the Otsu log level.
        """
        self.otsu_log_level = 10**self.threshold_otsu_positive(self.ID.image_raw, scaling=self.ID.scaling_options[self.ID.i_logarithmic])
        
    def calculate_average_level(self, threshold_val):
        """
        Calculates the average contour level for the raw image data based on a specified threshold value.
        """
        self.average_level = self.find_average_level(self.ID.image_raw, threshold_percentage=threshold_val)
    #-------------------

    #-------------------
    # Calculate Smoothed Levels
    #-------------------        
    def calculate_lcc_level_smoothed(self):
        """
        Calculates the Last Closed Contour (LCC) level for the smoothed image data. 
        Updates global levels, generates contour lines, and determines if they are closed.
        """
        self.update_global_levels_smoothed()
        self.CS_smoothed = plt.contour(self.ID.X, self.ID.Y, self.ID.image_smoothed, levels=self.global_levels_smoothed, linewidths=0)
        self.lcc_level_smoothed = self.is_contour_closed(self.ID.X, self.ID.Y, self.ID.image_smoothed, self.CS_smoothed, self.global_levels_smoothed, self.M.mask)
    
    def calculate_otsu_level_smoothed(self):
        """
        Calculates the Otsu threshold level for the smoothed image data and updates the Otsu level.
        """
        self.otsu_level_smoothed = self.threshold_otsu_positive(self.ID.image_smoothed)        
    
    def calculate_otsu_log_level_smoothed(self):
        """
        Calculates the Otsu threshold level for the logarithmically scaled smoothed image data 
        and updates the Otsu log level.
        """
        self.otsu_log_level_smoothed = 10**self.threshold_otsu_positive(self.ID.image_smoothed, scaling=self.ID.scaling_options[self.ID.i_logarithmic])
    
    def calculate_average_level_smoothed(self, threshold_val):
        """
        Calculates the average contour level for the smoothed image data based on a specified threshold value.
        """
        self.average_level_smoothed = self.find_average_level(self.ID.image_smoothed, threshold_percentage=threshold_val)
    #-------------------
    
    #-------------------
    # Calculate No Background Levels
    #-------------------    
    def calculate_lcc_level_nobackground(self):
        """
        Calculates the Last Closed Contour (LCC) level for the no-background image data. 
        Updates global levels, generates contour lines, and determines if they are closed.
        """
        self.update_global_levels_nobackground()
        self.CS_nobackground = plt.contour(self.ID.X, self.ID.Y, self.ID.image_nobackground, levels=self.global_levels_nobackground, linewidths=0)
        self.lcc_level_nobackground = self.is_contour_closed(self.ID.X, self.ID.Y, self.ID.image_nobackground, self.CS_nobackground, self.global_levels_nobackground, self.M.mask)
    
    def calculate_otsu_level_nobackground(self):
        """
        Calculates the Otsu threshold level for the no-background image data and updates the Otsu level.
        """
        self.otsu_level_nobackground = self.threshold_otsu_positive(self.ID.image_nobackground)
    
    def calculate_otsu_log_level_nobackground(self):
        """
        Calculates the Otsu threshold level for the logarithmically scaled no-background image data 
        and updates the Otsu log level.
        """
        self.otsu_log_level_nobackground = 10**self.threshold_otsu_positive(self.ID.image_nobackground, scaling=self.ID.scaling_options[self.ID.i_logarithmic])
   
    def calculate_average_level_nobackground(self, threshold_val):
        """
        Calculates the average contour level for the no-background image data based on a specified threshold value.
        """
        self.average_level_nobackground = self.find_average_level(self.ID.image_nobackground, threshold_percentage=threshold_val)
    #-------------------

    #-------------------
    # Calculate Level Combinations
    #-------------------  
    def calculate_raw_levels(self, threshold_val):
        """
        Calculates all contour levels (LCC, Otsu, Otsu Log, and Average) for the Raw image data.
        """
        self.calculate_lcc_level()
        self.calculate_otsu_level()
        self.calculate_otsu_log_level()
        self.calculate_average_level(threshold_val)
    
    def calculate_smoothed_levels(self, threshold_val):
        """
        Calculates all contour levels (LCC, Otsu, Otsu Log, and Average) for the Smoothed image data.
        """
        self.calculate_lcc_level_smoothed()
        self.calculate_otsu_level_smoothed()
        self.calculate_otsu_log_level_smoothed()
        self.calculate_average_level_smoothed(threshold_val)
    
    def calculate_nobackground_levels(self, threshold_val):
        """
        Calculates all contour levels (LCC, Otsu, Otsu Log, and Average) for the NoBackground image data.
        """
        self.calculate_lcc_level_nobackground()
        self.calculate_otsu_level_nobackground()
        self.calculate_otsu_log_level_nobackground()
        self.calculate_average_level_nobackground(threshold_val)
   
    def calculate_lcc_levels(self):
        """
        Calculates the LCC (Last Closed Contour) level for raw, smoothed, and no-background image data.
        """
        self.calculate_lcc_level()
        self.calculate_lcc_level_smoothed()
        self.calculate_lcc_level_nobackground()
        
    def calculate_otsu_levels(self):
        """
        Calculates the Otsu threshold level for raw, smoothed, and no-background image data.
        """
        self.calculate_otsu_level()
        self.calculate_otsu_level_smoothed()
        self.calculate_otsu_level_nobackground()
    
    def calculate_otsu_log_levels(self):
        """
        Calculates the Otsu log threshold level for raw, smoothed, and no-background image data.
        """
        self.calculate_otsu_log_level()
        self.calculate_otsu_log_level_smoothed()
        self.calculate_otsu_log_level_nobackground()
    
    def calculate_average_levels(self, threshold_val):
        """
        Calculates the average contour level for raw, smoothed, and no-background image data based on a specified threshold value.
        """
        self.calculate_average_level(threshold_val)
        self.calculate_average_level_smoothed(threshold_val)
        self.calculate_average_level_nobackground(threshold_val)

    def calculate_all_levels(self, threshold_val):
        """
        Calculates all contour levels (LCC, Otsu, Otsu Log, and Average) for raw, smoothed, and no-background image data.
        """
        self.calculate_raw_levels(threshold_val)
        self.calculate_smoothed_levels(threshold_val)
        self.calculate_nobackground_levels(threshold_val)
    #-------------------

    #-------------------
    # Make Raw Contours
    #-------------------
    def make_lcc_contour(self):
        """
        Creates a contour plot for the LCC (Last Closed Contour) level of the raw image data.
        """
        self.lcc_contour = plt.contour(self.ID.X, self.ID.Y, self.ID.image_raw, levels=[self.lcc_level], linewidths=self.ID.line_width, linestyles=self.ID.line_style[self.i_raw], colors=[self.ID.line_color[self.i_lcc]])        
        
    def make_otsu_contour(self):
        """
        Creates a contour plot for the Otsu threshold level of the raw image data.
        """
        self.otsu_contour = plt.contour(self.ID.X, self.ID.Y, self.ID.image_raw, levels=[self.otsu_level], linewidths=self.ID.line_width, linestyles=self.ID.line_style[self.i_raw], colors=[self.ID.line_color[self.i_otsu]])

    def make_otsu_log_contour(self):
        """
        Creates a contour plot for the Otsu log threshold level of the raw image data.
        """
        self.otsu_log_contour = plt.contour(self.ID.X, self.ID.Y, self.ID.image_raw, levels=[self.otsu_log_level], linewidths=self.ID.line_width, linestyles=self.ID.line_style[self.i_raw], colors=[self.ID.line_color[self.i_otsulog]])
      
    def make_average_contour(self):
        """
        Creates a contour plot for the Average level of the raw image data.
        """
        self.average_contour = plt.contour(self.ID.X, self.ID.Y, self.ID.image_raw, levels=[self.average_level], linewidths=self.ID.line_width, linestyles=self.ID.line_style[self.i_raw], colors=[self.ID.line_color[self.i_average]])
    #-------------------

    #-------------------
    # Make Smoothed Contours
    #-------------------    
    def make_lcc_contour_smoothed(self):
        """
        Creates a contour plot for the LCC (Last Closed Contour) level of the smoothed image data.
        """
        self.lcc_contour_smoothed = plt.contour(self.ID.X, self.ID.Y, self.ID.image_smoothed, levels=[self.lcc_level_smoothed], linewidths=self.ID.line_width, linestyles=self.ID.line_style[self.i_smoothed], colors=[self.ID.line_color[self.i_lcc]])
    
    def make_otsu_contour_smoothed(self):
        """
        Creates a contour plot for the Otsu threshold level of the smoothed image data.
        """
        self.otsu_contour_smoothed = plt.contour(self.ID.X, self.ID.Y, self.ID.image_smoothed, levels=[self.otsu_level_smoothed], linewidths=self.ID.line_width, linestyles=self.ID.line_style[self.i_smoothed], colors=[self.ID.line_color[self.i_otsu]])
    
    def make_otsu_log_contour_smoothed(self):
        """
        Creates a contour plot for the Otsu log threshold level of the smoothed image data.
        """
        self.otsu_log_contour_smoothed = plt.contour(self.ID.X, self.ID.Y, self.ID.image_smoothed, levels=[self.otsu_log_level_smoothed], linewidths=self.ID.line_width, linestyles=self.ID.line_style[self.i_smoothed], colors=[self.ID.line_color[self.i_otsulog]])

    def make_average_contour_smoothed(self):
        """
        Creates a contour plot for the Average level of the smoothed image data.
        """
        self.average_contour_smoothed = plt.contour(self.ID.X, self.ID.Y, self.ID.image_smoothed, levels=[self.average_level_smoothed], linewidths=self.ID.line_width, linestyles=self.ID.line_style[self.i_smoothed], colors=[self.ID.line_color[self.i_average]])
    #-------------------
        
    #-------------------
    # Make No Background Contours
    #-------------------    
    def make_lcc_contour_nobackground(self):
        """
        Creates a contour plot for the LCC (Last Closed Contour) level of the no-background image data.
        """
        self.lcc_contour_nobackground = plt.contour(self.ID.X, self.ID.Y, self.ID.image_nobackground, levels=[self.lcc_level_nobackground], linewidths=self.ID.line_width, linestyles=self.ID.line_style[self.i_nobackground], colors=[self.ID.line_color[self.i_lcc]])
    
    def make_otsu_contour_nobackground(self):
        """
        Creates a contour plot for the Otsu threshold level of the no-background image data.
        """
        self.otsu_contour_nobackground = plt.contour(self.ID.X, self.ID.Y, self.ID.image_nobackground, levels=[self.otsu_level_nobackground], linewidths=self.ID.line_width, linestyles=self.ID.line_style[self.i_nobackground], colors=[self.ID.line_color[self.i_otsu]])
    
    def make_otsu_log_contour_nobackground(self):
        """
        Creates a contour plot for the Otsu log threshold level of the no-background image data.
        """
        self.otsu_log_contour_nobackground = plt.contour(self.ID.X, self.ID.Y, self.ID.image_nobackground, levels=[self.otsu_log_level_nobackground], linewidths=self.ID.line_width, linestyles=self.ID.line_style[self.i_nobackground], colors=[self.ID.line_color[self.i_otsulog]])

    def make_average_contour_nobackground(self):
        """
        Creates a contour plot for the Average level of the no-background image data.
        """
        self.average_contour_nobackground = plt.contour(self.ID.X, self.ID.Y, self.ID.image_nobackground, levels=[self.average_level_nobackground], linewidths=self.ID.line_width, linestyles=self.ID.line_style[self.i_nobackground], colors=[self.ID.line_color[self.i_average]])
    #-------------------
    
    #-------------------
    # Make Contour Combinations
    #-------------------
    def make_raw_contours(self):
        """
        Creates contour plots for all contour levels (LCC, Otsu, Otsu Log, and Average) using Raw image data.
        """
        self.make_lcc_contour()
        self.make_otsu_contour()
        self.make_otsu_log_contour()
        self.make_average_contour()

    def make_smoothed_contours(self):
        """
        Creates contour plots for all contour levels (LCC, Otsu, Otsu Log, and Average) using Smoothed image data.
        """
        self.make_lcc_contour_smoothed()
        self.make_otsu_contour_smoothed()
        self.make_otsu_log_contour_smoothed()
        self.make_average_contour_smoothed()

    def make_nobackground_contours(self):
        """
        Creates contour plots for all contour levels (LCC, Otsu, Otsu Log, and Average) using NoBackground image data.
        """
        self.make_lcc_contour_nobackground()
        self.make_otsu_contour_nobackground()
        self.make_otsu_log_contour_nobackground()
        self.make_average_contour_nobackground()

    def make_lcc_contours(self):
        """
        Creates contour plots for the LCC (Last Closed Contour) level using raw, smoothed, and no-background image data.
        """
        self.make_lcc_contour()
        self.make_lcc_contour_smoothed()
        self.make_lcc_contour_nobackground()
    
    def make_otsu_contours(self):
        """
        Creates contour plots for the Otsu threshold level using raw, smoothed, and no-background image data.
        """
        self.make_otsu_contour()
        self.make_otsu_contour_smoothed()
        self.make_otsu_contour_nobackground()

    def make_otsu_log_contours(self):
        """
        Creates contour plots for the Otsu log threshold level using raw, smoothed, and no-background image data.
        """
        self.make_otsu_log_contour()
        self.make_otsu_log_contour_smoothed()
        self.make_otsu_log_contour_nobackground()
        
    def make_average_contours(self):
        """
        Creates contour plots for the Average level using raw, smoothed, and no-background image data.
        """
        self.make_average_contour()
        self.make_average_contour_smoothed()
        self.make_average_contour_nobackground()

    def make_all_contours(self):
        """
        Creates contour plots for all contour levels (LCC, Otsu, Otsu Log, and Average) using raw, smoothed, and no-background image data.
        """
        self.make_raw_contours()
        self.make_smoothed_contours()
        self.make_nobackground_contours()
    #-------------------

    #-------------------
    def initialise_all_contours(self):
        """
        Initializes all contours by creating global contour lines, calculating all levels based on the initial averaging threshold percentage, 
        creating all contour plots, setting up contour dictionaries, and updating contour levels.
        """
        self.make_global_contour_lines()
        self.calculate_all_levels(self.initial_averaging_threshold_percentage)
        self.make_all_contours()
        self.setup_contour_dictionaries()
        self.update_contour_levels()
    #-------------------

    #-------------------
    def get_next_contour_below(self, current_level, levels):
        """
        Finds the next contour level below the current level from a list of levels.
    
        Args:
            current_level (float): The current contour level.
            levels (list of float): A list of contour levels to search through.
    
        Returns:
            float or None: The next contour level below the current level, or None if no such level exists.
        """
        # Ensure levels are defined
        if levels is None:
            return None
    
        # Find the next contour level below the current value
        for level in reversed(levels):
            if level < current_level:
                return level
    
        return None
    #-------------------
#-------------------End ContourLevels


#***********************************************************
#===========================================================
#               Interactive Controls
#===========================================================
#***********************************************************


#-------------------
# Smoothing controls
#-------------------
class SmoothingControls:
    """
    A class to manage the interactive smoothing controls.
    """
    
    def __init__(self, S):
        """
        Initializes the interactive SmoothingControls instance.
        
        Args:
            S (object): An instance of the Smoothing (S) class.
        """
        self.S = S
        self.setup_widget()
        
    def setup_widget(self):
        """
        Setup of all of the smoothing parameter widgets.
        """
        import ipywidgets as widgets

        # Dictionary to store created widgets
        self.smoothing_widgets = {}

        # Loop through each smoothing parameter in the dictionary and setup a widget
        for method, params in self.S.smoothing_parameters.items():
            for param_index, param_info in params.items():
                # Extract important widget information
                var_name  = param_info['var_name']
                label     = param_info['label']
                value     = param_info['value']
                min_value = param_info.get('min_value', None)
                max_value = param_info.get('max_value', None)
                step      = param_info.get('step', None)
                options   = param_info.get('options', None)
        
                if method not in self.smoothing_widgets:
                    self.smoothing_widgets[method] = []

                # Select the appropriate widget type
                if options:
                    widget = Utilities.create_dropdown(label, options, value)
                elif step is not None:
                    widget = Utilities.create_float_text(label, value, min_value, max_value, step)
                else:
                    widget = Utilities.create_int_text(label, value, min_value, max_value)

                # Append the new widget to the list
                self.smoothing_widgets[method].append(widget)

        # Create a dropdown widget for selecting the smoothing method
        self.smoothing_dropdown = widgets.Dropdown(
            options     = self.S.smoothing_methods,
            value       = self.S.initial_smoothing_method,
            description = 'Smoothing Method:',
            style       = {'description_width': 'initial'}
        )

        # Combine the smoothing dropdown with its controls horizontally
        self.smoothing_controls = widgets.HBox([self.smoothing_dropdown, *self.smoothing_widgets[self.smoothing_dropdown.value]])
#-------------------End SmoothingControls


#-------------------
# CMap Data controls
#-------------------
class DataControls:
    """
    A class to manage the interactive colormap data controls.
    """
    
    def __init__(self, ID, CL):
        """
        Initializes the interactive DataControls instance.
        
        Args:
            ID (object): The ImageData (ID) object containing colormap and scaling options.
            CL (object): The ContourLines (CL) object containing data list and indices.
        """
        self.ID = ID
        self.CL = CL
        self.setup_widget()

    def setup_widget(self):
        """
        Sets up the interactive widgets for controlling colormap data.

        This method initializes various widgets for user interaction, including:
        - Dropdown for selecting the data to be visualized with a colormap.
        - Toggle buttons for choosing between linear and logarithmic scaling.
        - Dropdown for selecting the colormap.
        - FloatText inputs for setting the minimum and maximum limits of the colorbar.
        """
        import ipywidgets as widgets

        # Initial values for colorbar limits
        initial_min_step_size = Utilities.calculate_step_size(self.ID.cbar_min)
        initial_max_step_size = Utilities.calculate_step_size(self.ID.cbar_max)

        # Create dropdown menu for selecting background data
        self.dataSelector = widgets.Dropdown(
            options     = self.CL.dataList,
            value       = self.CL.dataList[self.CL.i_raw],
            description = 'Colormap Data:',
            style       = {'description_width': 'initial'}
        )
        self.dataSelector.layout.width = '200px'

        # Create toggle buttons for linear/log scaling
        self.scaling_toggle = widgets.ToggleButtons(
            options     = self.ID.scaling_options,
            value       = self.ID.scaling_options[self.ID.i_linear],
            description = '',
            disabled    = False,
            style       = {"button_width": "auto"}
        )

        # Get a list of all available colormaps
        colormap_options = plt.colormaps()

        # Create dropdown menu for selecting the colormap
        self.colormap_dropdown = widgets.Dropdown(
            options     = colormap_options,
            value       = self.ID.cmapname,  # Set default value
            description = ''
        )
        self.colormap_dropdown.layout.width = '100px'

        # Create input boxes for setting colorbar min limit
        self.clim_min_input = widgets.FloatText(
            value       = self.ID.cbar_min,
            description = 'Limits [Min,Max]:',
            style       = {'description_width': 'initial'},
            disabled    = False,
            step        = initial_min_step_size
        )
        self.clim_min_input.layout.width = '200px'

        # Create input boxes for setting colorbar max limit
        self.clim_max_input = widgets.FloatText(
            value       = self.ID.cbar_max,
            description = '',
            style       = {'description_width': 'initial'},
            disabled    = False,
            step        = initial_max_step_size
        )
        self.clim_max_input.layout.width = '100px'

        # Combine data menu, scaling buttons, and colormap dropdown
        self.data_controls = widgets.HBox([self.dataSelector, self.scaling_toggle, self.colormap_dropdown, self.clim_min_input, self.clim_max_input])
#-------------------End DataControls


#-------------------
# Checkbox controls
#-------------------
class CheckboxControls:
    """
    A class to manage the interactive checkbox controls for contour comparison.
    """
    
    def __init__(self, CL):
        """
        Initializes the interactive CheckboxControls instance.
        
        Args:
            CL (object): The Contour Lines (CL) object containing row labels, column labels, and contour lines.
        """
        self.CL = CL
        self.setup_widget()

    def display_checkbox_grid(self):
        """
        Creates a grid of checkboxes for contour comparison.

        This method generates a grid of checkboxes arranged according to the
        provided row and column labels from the CL object. Each row represents
        a different dataset, and each column represents a specific contour type.

        Returns:
            widgets.GridBox: A widget containing the checkbox grid.
        """
        import ipywidgets as widgets

        # Create an empty list to store rows of checkboxes
        checkboxes = []

        # Loop through each row label and its index in CL.row_labels
        for i, row_label in enumerate(self.CL.row_labels):
            # Create a list to store widgets for the current row
            row = [
                # Add a label widget for the row label, right-aligned with auto width
                widgets.Label(value=row_label, layout=widgets.Layout(justify_content='flex-end', width='auto'))
            ]
            # Add checkboxes for each column, except the first one which contains only labels
            row += [
                widgets.Checkbox(
                    value=(i == 0),  # Check the first checkbox in each row, uncheck others
                    description='',  # No description for checkboxes
                    layout=widgets.Layout(width='auto', margin='0px 0px', padding='0px 0px')  # Auto width with no margin/padding
                ) for _ in self.CL.column_labels[1:]
            ]
            # Append the row of widgets to the list of rows
            checkboxes.append(row)
        
        # Insert column labels as the first row of checkboxes
        checkboxes.insert(0, [
            # Create label widgets for each column label, right-aligned with auto width
            widgets.Label(value=col_label, layout=widgets.Layout(justify_content='flex-end', width='auto')) 
            for col_label in self.CL.column_labels
        ])
        
        # Create a grid box widget to contain all checkboxes
        grid = widgets.GridBox(
            children=sum(checkboxes, []),  # Flatten the list of lists into a single list of widgets
            layout=widgets.Layout(
                grid_template_columns='auto ' * len(self.CL.column_labels),  # Auto-sized columns based on column count
                justify_content='flex-end'  # Right-align the grid within its container
            )
        )

        return grid

    def setup_widget(self):
        """
        Sets up the interactive widgets for controlling checkboxes.

        This method initializes various widgets for user interaction, including:
        - A grid of checkboxes for selecting contours.
        - Buttons for checking/unchecking all checkboxes.
        - Labels for organizing and describing the controls.
        """
        import ipywidgets as widgets

        # Calculate the number of rows and columns in the checkbox grid
        self.num_rows = len(self.CL.row_labels)
        self.num_cols = len(self.CL.column_labels)
        
        # Calculate the indices of checkboxes for different contour types
        self.i_average_checkmarks = self.checkbox_index([m * (self.num_cols - 1) - 1 for m in range(1, self.num_cols - 1, 1)])
        self.i_smoothed_checkmarks = self.checkbox_index([(self.num_cols - 1) - 1 + m for m in range(1, self.num_cols, 1)])
        self.i_nobackground_checkmarks = self.checkbox_index([2 * (self.num_cols - 1) - 1 + m for m in range(1, self.num_cols, 1)])
        
        # Create the checkbox grid
        self.checkboxes_grid = self.display_checkbox_grid()
        
        # Create a vertical box to hold the checkbox grid with layout settings
        self.checkboxes_box = widgets.VBox([self.checkboxes_grid], layout=widgets.Layout(align_items='flex-start', spacing=5))
        
        # Create a label widget for comparing contours with layout settings
        self.compare_label = widgets.Label(value='Compare Contours:', layout=widgets.Layout(width='120px', margin='0px 0px 0px 0px'))
        
        # Create label widgets for spacing and buttons for checking/unchecking all checkboxes
        self.empty_label = widgets.Label(value=' ', layout=widgets.Layout(height='20px'))
        self.check_all_button = widgets.Button(description='Check All', layout={"width": "100px"})
        self.uncheck_all_button = widgets.Button(description='Uncheck All', layout={"width": "100px"})
        
        # Create a vertical box to hold buttons with layout settings
        self.buttons_vbox = widgets.VBox([self.empty_label, self.check_all_button, self.uncheck_all_button])
        
        # Create a horizontal box to hold the comparison label/buttons box and the checkbox grid
        self.checkbox_controls = widgets.HBox([widgets.VBox([self.compare_label, self.buttons_vbox]), self.checkboxes_box])

    def checkbox_index(self, i):
        """
        Calculates the index of a checkbox in the grid, accounting for labels in the first row and column.

        Args:
            i (int or list of int): The index or indices of the checkboxes in the logical grid.

        Returns:
            int or list of int: The actual index or indices in the flattened grid.
        """
        # The checkbox grid is complicated by the labels, which need to be skipped.
        if isinstance(i, list):
            # Calculate checkbox index for each element in the list
            index = [((j // (self.num_cols - 1)) * self.num_cols + j % (self.num_cols - 1) + self.num_cols + 1) for j in i]
        else:
            # Calculate checkbox index for a single element
            index = ((i // (self.num_cols - 1)) * self.num_cols + i % (self.num_cols - 1) + self.num_cols + 1)
        return index

    def are_lines_visible(self, selected_data=None):
        """
        Determines the visibility of contour lines based on checkbox selections.

        Args:
            selected_data (str, optional): The specific dataset to check visibility for. Defaults to None.

        Returns:
            list of bool: A list indicating the visibility of each contour line.
        """
        visible_values = []
        checkbox_index = 0  # Initialize checkbox index counter
        for key, sublist in self.CL.contour_lines.items():  # Use .items() to access both keys and values
            for line in sublist:
                if selected_data is None or key == selected_data:
                    visible_values.append(self.checkboxes_grid.children[self.checkbox_index(checkbox_index)].value)
                checkbox_index += 1  # Increment the checkbox index counter

        return visible_values
#-------------------End CheckboxControls


#-------------------
# Averaging controls
#-------------------
class AveragingControls:
    """
    A class to manage the interactive averaging controls.
    """
    
    def __init__(self, CL):
        """
        Initializes the interactive AveragingControls instance.
        
        Args:
            CL (object): The Contour Lines (CL) instance.
        """
        self.CL = CL
        self.setup_widget()

    def setup_widget(self):
        """
        Set up the interactive widget for adjusting the threshold percentage.
        """
        import ipywidgets as widgets

        # Create the widget label
        self.averaging_label = widgets.Label(value='Averaging Parameters:', layout={'margin': '0px 0px 0px 0px'})
    
        # Create a float text widget for setting the threshold percentage with specified settings
        self.averaging_threshold_percentage = Utilities.create_float_text('Threshold %', value=self.CL.initial_averaging_threshold_percentage, min_value=0, max_value=99.99, step=1, width='150px')
    
        # Store the label and widget side-by-side
        self.averaging_controls = widgets.HBox([self.averaging_label, self.averaging_threshold_percentage])
#-------------------End AveragingControls


#-------------------
# Remove Background controls
#-------------------
class RemoveBackgroundControls:
    """
    A class to manage the interactive controls for removing background from an image.
    """
    
    def __init__(self, NB):
        """
        Initializes the interactive RemoveBackgroundControls instance.

        Args:
            NB (object): The NoBackground (NB) instance.
        """
        self.NB = NB
        self.setup_widget()

    def setup_widget(self):
        """
        Set up the interactive widget for adjusting background removal parameters.

        - Filter shape: The shape of the filter used for background removal. Options include Gaussian, Uniform, and Median.
        - Filter size: The size of the filter, which determines the area over which background will be removed.
        - Pad mode: The padding mode used for edges. Options include Reflect, Constant, and Wrap.
        - Pad value: The value used for padding to the edges of the input when the pad mode is set to Constant.
        """
        import ipywidgets as widgets

        # Create a label for the background removal section
        self.remove_background_label = widgets.Label(value='Remove Background:  ', layout={'margin': '0px 0px 0px 0px'})

        # Create dropdown menus and text boxes for background removal parameters
        self.filter_shape_dropdown = widgets.Dropdown(**self.NB.nobackground_parameters[self.NB.i_filter_shape])
        self.filter_size_text      = Utilities.create_int_text(**self.NB.nobackground_parameters[self.NB.i_filter_size])
        self.pad_mode_dropdown     = widgets.Dropdown(**self.NB.nobackground_parameters[self.NB.i_pad_mode])
        self.pad_value             = Utilities.create_float_text(**self.NB.nobackground_parameters[self.NB.i_pad_value])

        # Create a dictionary to store the widgets
        self.nobackground_widgets = {
            self.NB.i_filter_shape: self.filter_shape_dropdown,
            self.NB.i_filter_size:  self.filter_size_text,
            self.NB.i_pad_mode:     self.pad_mode_dropdown,
            self.NB.i_pad_value:    self.pad_value
        }

        # Store widgets together in a horizontal box (only include pad_value widget if pad_mode == 'constant')
        if self.NB.initial_pad_mode == self.NB.padding_options[self.NB.i_constant]:
            self.remove_background_controls = widgets.HBox([self.remove_background_label, self.filter_shape_dropdown, self.filter_size_text, self.pad_mode_dropdown, self.pad_value])
        else:
            self.remove_background_controls = widgets.HBox([self.remove_background_label, self.filter_shape_dropdown, self.filter_size_text, self.pad_mode_dropdown])
#-------------------End RemoveBackgroundControls


#-------------------
# Mask Data controls
#-------------------
class MaskDataControls:
    """
    A class to manage interactive controls for masking data.
    """
    def __init__(self, ID, M):
        """
        Initializes the MaskDataControls instance.

        Args:
            ID (object): The instance containing Image Data (ID).
            M (object): The instance containing Mask (M) data.
        """
        self.ID = ID
        self.M  = M
        self.setup_widgets()

    def setup_widgets(self):
        """
        Initializes the interactive widget for controlling data masking.

        - Mask Data: Dropdown menu to toggle masking on or off.
        - Mask Shape: Dropdown menu to select the shape of the mask, such as circle, square, etc.
        - Size: Integer text box to specify the size of the mask.
        - Center (x,y): FloatText inputs for setting the center coordinates of the mask.
        """
        import ipywidgets as widgets

        self.mask_data_label = widgets.Label(value='Mask Data:  ', layout={'margin': '0px 0px 0px 0px'})
        self.mask_yes_no_dropdown = widgets.Dropdown(options=['Yes', 'No'], value='No', layout={'width': '50px'})
        self.mask_shape_dropdown = widgets.Dropdown(description='Mask Shape', options=self.M.mask_shape_options, value=self.M.initial_mask_shape, layout={'width': '200px'})
        self.mask_size_text = Utilities.create_int_text('Size', value=self.M.initial_mask_size, min_value=1, max_value=max(len(self.ID.x), len(self.ID.y)))

        # Create input boxes for setting the x position of the mask centre
        self.mask_xpos_text = widgets.FloatText(
            value=self.M.initial_mask_xpos,  # Set default value
            description=' Center (x,y):',
            disabled=False,
            step=self.M.initial_x_step_size)
        self.mask_xpos_text.layout.width = '190px'

        # Create input boxes for setting the y position of the mask centre
        self.mask_ypos_text = widgets.FloatText(
            value=self.M.initial_mask_ypos,  # Set default value
            description='',
            disabled=False,
            step=self.M.initial_y_step_size)
        self.mask_ypos_text.layout.width = '100px'

        # Function to update the options visibility based on the yes/no dropdown value
        def update_mask_options_visibility(change):
            visibility = 'visible' if change['new'] == 'Yes' else 'hidden'
            for widget in [self.mask_shape_dropdown, self.mask_size_text, self.mask_xpos_text, self.mask_ypos_text]:
                widget.layout.visibility = visibility

        # Set options visibility based on the default value of the yes/no dropdown
        update_mask_options_visibility({'new': self.mask_yes_no_dropdown.value})

        # Attach the update function to the yes/no dropdown
        self.mask_yes_no_dropdown.observe(update_mask_options_visibility, names='value')

        self.mask_data_controls = widgets.HBox([self.mask_data_label, self.mask_yes_no_dropdown, self.mask_shape_dropdown, self.mask_size_text, self.mask_xpos_text, self.mask_ypos_text])
#-------------------End MaskDataControls


#-------------------
# Print Data controls
#-------------------
class PrintData:
    """
    A class to manage printing results and code in the interactive session.
    """
    
    def __init__(self, ID, M, CL, MDC, SC, DC, CC, AC, interface_name, file_name, get_smoothing_parameters, get_nobackground_params, custom_mask=None):
        """
        Initializes the PrintData instance.

        Args:
            ID  (object): The instance of the Image Data (ID) class.
            M   (object): The instance of the Mask (M) class.
            CL  (object): The instance of the Contour Levels (CL) class.
            MDC (object): The instance of the Mask Data Controls (MDC) class.
            SC  (object): The instance of the Smoothing Controls (SC) class.
            DC  (object): The instance of the Data Controls (DC) class.
            CC  (object): The instance of the Checkbox Controls (CC) class.
            AC  (object): The instance of the Averaging Controls (AC) class.
            interface_name: The name of the user interface.
            get_smoothing_parameters: A function to get the smoothing parameters (defined in the UserInterface class).
            get_nobackground_params: A function to get the parameters for removing background (defined in the UserInterface class).
            custom_mask: A custom mask if provided.
        """
        import ipywidgets as widgets

        self.ID  = ID
        self.M   = M
        self.CL  = CL
        self.MDC = MDC
        self.SC  = SC
        self.DC  = DC
        self.CC  = CC
        self.AC  = AC
        self.interface_name = interface_name
        self.file_name = file_name
        self.get_smoothing_parameters = get_smoothing_parameters
        self.get_nobackground_params  = get_nobackground_params
        self.custom_mask = custom_mask
        
        # Create a button widget to toggle the display of information
        self.toggle_button = widgets.Button(description="Toggle Info Display")
        
        # Create an Output widget to display the information
        self.output = widgets.Output()
        
        # Variable to track whether the information is currently displayed
        self.info_displayed = False
        
        # Attach the toggle_info function to the button's 'on_click' event
        self.toggle_button.on_click(self.toggle_info)
        
    def toggle_info(self, button):
        """
        Toggles the display of level information and executable code in the interactive plot.

        Args:
            button: The button widget.
        """
        from IPython.display import clear_output

        with self.output:
            clear_output(wait=True)  # Clear previous output
            if not self.info_displayed:
                # Print the information here
                self.interactive_print()
                self.info_displayed = True
            else:
                # Clear the output to remove the printed information
                self.output.clear_output()
                self.info_displayed = False
                
    def display_results(self):
        """
        Display the toggle button and output widgets
        """
        from IPython.display import display

        display(self.toggle_button)
        display(self.output)

    def color_text_red(self, text):
        """
        Color text red
        """
        return f"\033[91m{text}\033[0m"

    def color_text_green(self, text):
        """
        Color text green
        """
        return f"\033[92m{text}\033[0m"

    def print_executable_code(self):
        """
        Generate executable Python code based on the current settings and user interactions.
    
        This method extracts visible contour levels and generates executable code snippets
        for reproducing the contour plots programmatically.  
        """
        # Determine which contours are visible in the interactive plot (i.e. which checkboxes are ticked)
        visible_flags_raw          = self.CC.are_lines_visible(self.CL.dataList[self.CL.i_raw])
        visible_flags_smoothed     = self.CC.are_lines_visible(self.CL.dataList[self.CL.i_smoothed])
        visible_flags_nobackground = self.CC.are_lines_visible(self.CL.dataList[self.CL.i_nobackground])

        # Extract the visible levels
        levels_raw          = [level for level, visible in zip(self.CL.contour_levels[self.CL.dataList[self.CL.i_raw]], visible_flags_raw) if visible]
        levels_smoothed     = [level for level, visible in zip(self.CL.contour_levels[self.CL.dataList[self.CL.i_smoothed]], visible_flags_smoothed) if visible]
        levels_nobackground = [level for level, visible in zip(self.CL.contour_levels[self.CL.dataList[self.CL.i_nobackground]], visible_flags_nobackground) if visible]

        # Assign an instance name for the UserInterface in the executable code
        instance_name = 'CAT'

        # Color the items red that the user will need to change
        image_text    = self.color_text_red('[image]')
        extent_text   = self.color_text_red('[extent]')

        # Get the text to handle the mask
        if self.custom_mask is None:
            if self.MDC.mask_yes_no_dropdown.value == 'Yes':
                # Extract mask variables and values from widgets
                mask_xpos  = self.MDC.mask_xpos_text.value
                mask_ypos  = self.MDC.mask_ypos_text.value
                mask_shape = self.MDC.mask_shape_dropdown.value
                mask_size  = self.MDC.mask_size_text.value
    
                # Format and print the values for the create_mask function
                mask_text = f"{instance_name}.M.create_mask(mask_xpos={mask_xpos}, mask_ypos={mask_ypos}, mask_shape='{mask_shape}', mask_size={mask_size})"
            else:
                mask_text = 'None'
        else:
            mask_text = self.color_text_red('[mask]')

        # Set the scaling option for executable code
        if self.DC.scaling_toggle.value == self.ID.scaling_options[self.ID.i_logarithmic]:
            scaling = self.ID.scaling_options[self.ID.i_logarithmic]
        else:
            scaling = None

        # Set base names for the return variables in the executable code
        output_vars = "level_dict, background_image, plot"

        if any(self.CC.are_lines_visible()):
            print('')
            print('--------------------------------------------------------------------------')
            print('Copy and run the following code to reproduce the contours programatically:')
            print(' (red text must first be replaced with the appropriate input variables)')
            print('--------------------------------------------------------------------------')
            print(self.color_text_green('# Setup User Interface'))
            print(f"{instance_name} = {self.file_name}.{self.interface_name}({image_text}, {extent_text}, interactive=False)")

        # If Raw contours are visible, print executable code to produce the raw image and contours
        if any(visible_flags_raw):
            suffix = f"_{self.CL.dataList[self.CL.i_raw].lower().replace(' ', '')}"
            output_vars_with_suffix = ', '.join([f"{var}{suffix}" for var in output_vars.split(', ')])
            print(self.color_text_green('# Raw Image Contours'))
            print(f"{output_vars_with_suffix} = {instance_name}.find_contours_raw(threshold={self.AC.averaging_threshold_percentage.value}, selected_levels={levels_raw}, selected_scaling={scaling}, mask={mask_text})")

        # If Smoothed contours are visible, print executable code for the smoothed image and contours
        if any(visible_flags_smoothed):
            selected_method, smoothing_params = self.get_smoothing_parameters(self.SC.smoothing_dropdown.value)
            smoothing_params_str = ", ".join([f"{key}='{value}'" if isinstance(value, str) else f"{key}={value}" for key, value in smoothing_params.items()])

            suffix = f"_{self.CL.dataList[self.CL.i_smoothed].lower().replace(' ', '')}"
            output_vars_with_suffix = ', '.join([f"{var}{suffix}" for var in output_vars.split(', ')])
            print(self.color_text_green('# Smoothed Image Contours'))
            print(f"{output_vars_with_suffix} = {instance_name}.find_contours_smoothed(selected_method='{selected_method}', threshold={self.AC.averaging_threshold_percentage.value}, selected_levels={levels_smoothed}, selected_scaling={scaling}, {smoothing_params_str}, mask={mask_text})")

        # If NoBackground contours are visible, print executable code for the no-background image and contours
        if any(visible_flags_nobackground):
            nobackground_params = self.get_nobackground_params()
            nobackground_params_str = ", ".join([f"{key}='{value}'" if isinstance(value, str) else f"{key}={value}" for key, value in nobackground_params.items()])

            suffix = f"_{self.CL.dataList[self.CL.i_nobackground].lower().replace(' ', '')}"
            output_vars_with_suffix = ', '.join([f"{var}{suffix}" for var in output_vars.split(', ')])
            print(self.color_text_green('# No Background Image Contours'))
            print(f"{output_vars_with_suffix} = {instance_name}.find_contours_nobackground(threshold={self.AC.averaging_threshold_percentage.value}, selected_levels={levels_nobackground}, selected_scaling={scaling}, {nobackground_params_str}, mask={mask_text})")

    def interactive_print(self):
        """
        Items to print every time the interactive display is updated
        """
        self.print_levels()
        self.print_executable_code()

    def print_levels(self):
        """
        Print the actual level values for each visible contour in the interactive display.
        Also listed are the level spacings next to each data type (Raw, Smoothed, NoBackground)
        """
        # Iterate over each key-value pair in the contour levels dictionary
        for i, (key, sub_dict) in enumerate(self.CL.contour_levels.items()):
            # Find the maximum length of the sub-dictionary keys
            max_sub_key_length = max(len(sub_key) for sub_key in sub_dict.keys())
            
            # Check if any sub-level within the current key is visible
            sub_level_visible = any(
                # Check if the checkbox corresponding to the sub-level is checked
                self.CC.checkboxes_grid.children[self.CC.checkbox_index(i * len(self.CL.delta_levels) + index + i)].value
                for index, (_, _) in enumerate(sub_dict.items())
            )
            
            # If any sub-level is visible, print the key and its corresponding delta level
            if sub_level_visible:
                print(f"{key}: delta_levels = {self.CL.delta_levels[i]}")
                
                # Iterate over each sub-key and sub-value in the sub-dictionary
                for index, (sub_key, sub_value) in enumerate(sub_dict.items()):
                    # Check if the checkbox corresponding to the sub-level is checked
                    if self.CC.checkboxes_grid.children[self.CC.checkbox_index(i * len(self.CL.delta_levels) + index + i)].value:
                        # Determine the value to print based on whether it's a tuple or not
                        if isinstance(sub_value, tuple):
                            value = sub_value[0]  # Take the first element of the tuple
                        else:
                            value = sub_value
                        
                        # Print the sub-key and its corresponding value, left-aligned for clarity
                        print(f"    {sub_key.ljust(max_sub_key_length)}: {value}")

#-------------------End PrintData


#-------------------
# Interactive Observers and Update Functions
#-------------------
class InteractivePlot:
    """
    A class to manage interactive plotting and widget updates.
    """
    
    def __init__(self, ID, S, NB, M, CL, DC, CC, AC, SC, RBC, MDC, PD, custom_mask=None):
        """
        Initializes the InteractivePlot instance.

        Args:
            ID  (object): The instance of the Image Data (ID) class.
            S   (object): The instance of the Smoothing Controls (SC) class.
            NB  (object): The instance of the No Background (NB) class.
            M   (object): The instance of the Mask (M) class.
            CL  (object): The instance of the Contour Levels (CL) class.
            DC  (object): The instance of the Data Controls (DC) class.
            CC  (object): The instance of the Checkbox Controls (CC) class.
            AC  (object): The instance of the Averaging Controls (AC) class.
            SC  (object): The instance of the Smoothing Controls (SC) class.
            RBC (object): The instance of the Remove Background Controls (RBC) class.
            MDC (object): The instance of the Mask Data Controls (MDC) class.
            PD  (object): The instance of the Print Data (PD) class.
            custom_mask: A custom mask if provided.
        """
        import ipywidgets as widgets
        from IPython.display import display

        self.ID  = ID
        self.S   = S
        self.NB  = NB
        self.M   = M
        self.CL  = CL
        self.DC  = DC
        self.CC  = CC
        self.AC  = AC
        self.SC  = SC
        self.RBC = RBC
        self.MDC = MDC
        self.PD  = PD
        self.custom_mask = custom_mask

        Utilities.set_matplotlib_backend()

        # Initialise the figure
        self.ID.initialise_figure(self.CL)

        # Process the smoothed and no-background images
        self.S.initialise_image_smoothed()
        self.NB.initialise_image_nobackground()

        # Initialise all contours
        self.CL.initialise_all_contours()

        # Create a legend for the contours that can be dynamically updated
        self.legend = self.create_legend()

        # Update the contour visibility to just the default levels
        self.update_contour_visibility()

        #-------------------
        # Display Widgets, Figure, and Data
        #-------------------
        # Define the layout using widgets
        if self.custom_mask is None:
            self.main_layout = widgets.VBox([self.DC.data_controls, self.CC.checkbox_controls, self.AC.averaging_controls, self.SC.smoothing_controls, self.RBC.remove_background_controls, self.MDC.mask_data_controls])
        else:
            self.main_layout = widgets.VBox([self.DC.data_controls, self.CC.checkbox_controls, self.AC.averaging_controls, self.SC.smoothing_controls, self.RBC.remove_background_controls])
            
        # Display interactive widgets
        display(self.main_layout)
        
        # Display the figure
        plt.show()

        # Display printed values and code
        self.PD.display_results()
        #-------------------

        #-------------------
        # Data Observers
        #-------------------
        # Attach event handler to input box changes
        self.DC.clim_min_input.observe(lambda change: Utilities.update_min_value(self.DC.clim_min_input, self.DC.clim_max_input, self.ID.image_min), names='value')
        self.DC.clim_max_input.observe(lambda change: Utilities.update_max_value(self.DC.clim_min_input, self.DC.clim_max_input, self.ID.image_max), names='value')
        self.DC.clim_min_input.observe(lambda change: Utilities.update_step_size(self.DC.clim_min_input, self.DC.clim_max_input, change), names='value')
        self.DC.clim_max_input.observe(lambda change: Utilities.update_step_size(self.DC.clim_min_input, self.DC.clim_max_input, change), names='value')
        self.DC.clim_min_input.observe(self.on_clim_change, names='value')
        self.DC.clim_max_input.observe(self.on_clim_change, names='value')

        # Flag to indicate if widgets are being updated programmatically
        self.updating_widgets = False
        
        def update_widget_values_based_on_colorbar(self):
            """
            Update the widget values when the colorbar limits are changed interactively.
            """
            # Set the flag to indicate widgets are being updated
            self.updating_widgets = True
            try:
                # Temporarily disconnect observers to prevent unwanted updates
                self.DC.clim_min_input.unobserve(self.on_clim_change, names='value')
                self.DC.clim_max_input.unobserve(self.on_clim_change, names='value')

                # Ensure limits displayed in the widget are linear
                if self.DC.scaling_toggle.value == self.ID.scaling_options[self.ID.i_logarithmic]:
                    min_val = 10**(self.ID.background_cmap.get_clim()[0])
                    max_val = 10**(self.ID.background_cmap.get_clim()[1])
                else:
                    min_val = self.ID.background_cmap.get_clim()[0]
                    max_val = self.ID.background_cmap.get_clim()[1]

                # Update widget values based on colorbar limits
                self.DC.clim_min_input.value = min_val
                self.DC.clim_max_input.value = max_val
            finally:
                # Reconnect observers
                self.DC.clim_min_input.observe(self.on_clim_change, names='value')
                self.DC.clim_max_input.observe(self.on_clim_change, names='value')
                # Reset the flag
                self.updating_widgets = False

        # Ensure that changes to the colorbar limits update the widgets
        def on_colorbar_changed(self, event):
            """
            Callback function to update widget values when the colorbar changes.
            """
            update_widget_values_based_on_colorbar(self)

        # Connect the 'draw_event' of the canvas to the on_colorbar_changed function,
        self.ID.hF.canvas.mpl_connect('draw_event', lambda event: on_colorbar_changed(self, event))

        # Attach the update function to the dropdown's value change
        self.DC.dataSelector.observe(self.update_plot, names='value')
        
        # Attach the update function to the toggle switch's value change
        self.DC.scaling_toggle.observe(self.update_plot, names='value')
        
        # Attach the update function to the dropdown's value change
        self.DC.colormap_dropdown.observe(self.update_plot, names='value')
        #-------------------

        #-------------------
        # Mask Data Observers
        #-------------------
        # Attach update_figure function to the observe method of each control
        self.MDC.mask_yes_no_dropdown.observe(self.update_masked_figure, names='value')
        self.MDC.mask_xpos_text.observe(self.update_masked_figure, names='value')
        self.MDC.mask_ypos_text.observe(self.update_masked_figure, names='value')
        self.MDC.mask_shape_dropdown.observe(self.update_masked_figure, names='value')
        self.MDC.mask_size_text.observe(self.update_masked_figure, names='value')
        
        # Explicitly trigger the mask update if the initial value is 'Yes'
        if self.custom_mask is not None:
            self.MDC.mask_yes_no_dropdown.value = 'Yes'
            
        if self.MDC.mask_yes_no_dropdown.value == 'Yes':
            self.update_masked_figure({'new': 'Yes'})
        #-------------------
    
        #-------------------
        # Checkbox Observers
        #-------------------
        # Attach the check_all and uncheck_all functions to buttons' click events
        self.CC.check_all_button.on_click(lambda change:   self.check_all(change,   self.CC.checkboxes_grid))
        self.CC.uncheck_all_button.on_click(lambda change: self.uncheck_all(change, self.CC.checkboxes_grid))
        
        # Attach the update function to checkboxes' value change
        for i, item in enumerate(self.CC.checkboxes_grid.children):
            if isinstance(item, widgets.Checkbox):
                item.observe(self.update_plot, names='value')
        #-------------------

        #-------------------
        # Smoothing Observers
        #-------------------
        # Attach the update_smoothing_layout function to the dropdown's value change
        self.SC.smoothing_dropdown.observe(self.update_smoothing_layout,    names='value')
        self.SC.smoothing_dropdown.observe(self.update_smoothing, names='value')
        
        for method_name, widget_list in self.SC.smoothing_widgets.items():
            for widget in widget_list:
                widget.observe(self.update_smoothing, names='value')
        
        self.SC.smoothing_controls = widgets.HBox([self.SC.smoothing_dropdown, *self.SC.smoothing_widgets[self.SC.smoothing_dropdown.value]])
        #-------------------
        
        #-------------------
        # Averaging Observers
        #-------------------
        # Update threshold_val and recalculate contours when the value of averaging_threshold_percentage changes
        self.AC.averaging_threshold_percentage.observe(self.update_threshold_value, names='value')
        #-------------------
    
        #-------------------
        # Remove Background Observers
        #-------------------
        # Set up observer for the pad mode dropdown value
        self.RBC.nobackground_widgets[self.NB.i_pad_mode].observe(self.update_nobackground_layout, names='value')
        self.nobackground_widgets_to_observe = list(self.RBC.nobackground_widgets.values())
    
        for widget in self.nobackground_widgets_to_observe:
            widget.observe(self.update_nobackground, names='value')
        #-------------------

        #-------------------
        # Print Data Observers
        #-------------------
        # Observe changes in the scaling dropdown
        self.DC.scaling_toggle.observe(self.update_printed_info, names='value')

        # Observe changes in the mask dropdown
        self.MDC.mask_yes_no_dropdown.observe(self.update_printed_info, names='value')
        
        # Observe changes in the smoothing dropdown
        self.SC.smoothing_dropdown.observe(self.update_printed_info, names='value')
        
        # Observe changes in the smoothing widgets
        for method_name, widget_list in self.SC.smoothing_widgets.items():
            for widget in widget_list:
                widget.observe(self.update_printed_info, names='value')
        
        # Observe changes in the averaging threshold percentage
        self.AC.averaging_threshold_percentage.observe(self.update_printed_info, names='value')
        
        # Observe changes in the widgets related to remove background
        for widget in self.nobackground_widgets_to_observe:
            widget.observe(self.update_printed_info, names='value')
        
        # Observe changes in the checkboxes
        for i, item in enumerate(self.CC.checkboxes_grid.children):
            if isinstance(item, widgets.Checkbox):
                item.observe(self.update_printed_info, names='value')
        #-------------------


    #-------------------
    # Update Printed Info
    #-------------------
    def update_printed_info(self, change):
        """
        Updates the printed information when changes are made.
        """
        from IPython.display import clear_output

        # Update the printed line with the new slope value if it was previously displayed
        if self.PD.info_displayed:
            with self.PD.output:
                self.PD.output.clear_output()
                self.PD.interactive_print()
    #-------------------
                
    #-------------------
    # Create a Legend 
    #-------------------
    def create_legend(self):
        """
        Creates a legend based on the visible contour lines.
        """
        from matplotlib.lines  import Line2D

        # Initialize lists to store handles and labels for legend
        handles = []
        labels = []

        # Define the maximum number of contours per row
        max_contours_per_row = 4

        # Iterate through the contour lines dictionary
        for dataset_name, dataset_contours in self.CL.contour_lines.items():
            for contour_name, contour_object in dataset_contours.items():
                # Determine the index of the contour in the flattened list
                i = list(self.CL.contour_lines.keys()).index(dataset_name) * len(dataset_contours) + list(dataset_contours.keys()).index(contour_name)
                
                # Check if the contour is visible
                if self.CC.checkboxes_grid.children[self.CC.checkbox_index(i)].value:
                    # Get linestyle and color from contour_styles and contour_colors dictionaries based on contour name
                    linestyle = self.CL.contour_styles[dataset_name]['linestyle']
                    color = self.CL.contour_colors[contour_name]['color']
                    
                    # Create a Line2D object for the legend entry
                    handles.append(Line2D([0], [0], linestyle=linestyle, linewidth=2, color=color))
                    
                    # Create the label using contour name and dataset name
                    label = f"{contour_name} ({dataset_name})"
                    labels.append(label)

        # Reorder the handles and labels for the legend
        reorder = lambda handles_labels, num_columns: (sum((group[i::num_columns] for i in range(num_columns)), []) for group in handles_labels)
        handles_reordered, labels_reordered = reorder([handles, labels], max_contours_per_row)

        # Create the legend with the reordered handles and labels
        legend = self.ID.hA.legend(handles=handles_reordered, labels=labels_reordered, loc='upper center', bbox_to_anchor=(0.5, 1.165), ncol=max_contours_per_row, frameon=True, fancybox=True, shadow=True, fontsize=6)

        return legend

    def update_legend(self):
        """
        Removes the old legend and recreates it from scratch.
        """
        self.legend.remove()
        self.legend = self.create_legend()
        plt.draw()
    #-------------------

    #-------------------
    # Update Plot 
    #-------------------
    def update_plot(self, change):
        """
        Updates the background colormap and contour visibiility when necessary.
        """
        self.update_cmap()
        self.update_contour_visibility()
    #-------------------

    #-------------------
    # Update Averaging 
    #-------------------
    def update_threshold_value(self, change):
        """
        Updates the threshold value for averaging.
        """
        # Store the changed value
        threshold_val = change.new
        
        # Calculate the new average levels and update the contours
        self.CL.calculate_average_levels(threshold_val)
        self.update_contours(self.M.mask, update_levelList_set=self.CL.levelList[self.CL.i_average])
    #-------------------
    
    #-------------------
    # Update Contours
    #-------------------
    def update_contours(self, mask, update_dataList_set=None, update_levelList_set=None):
        """
        Updates selected contours based on widget values and the provided mask, including visibility.
        """
        # Update the calculated level values and contour lines
        self.CL.update_contour_levels()
        self.CL.update_contour_lines(mask, update_dataList_set=update_dataList_set, update_levelList_set=update_levelList_set)
        
        # Update the contour visibility based on the checkboxes
        self.update_contour_visibility()
    
    def update_contour_visibility(self):
        """
        Updates the visibility of contour lines based on checkbox settings.
        """
        # Flatten contour_lines into a single list of contour objects
        all_contours = [contour_object for dataset_contours in self.CL.contour_lines.values() for contour_object in dataset_contours.values()]
        
        # Enumerate over all_contours
        for i, contour_object in enumerate(all_contours):
            contour_object.set_visible(self.CC.checkboxes_grid.children[self.CC.checkbox_index(i)].value)

        # Update the legend and redraw
        self.update_legend()
        plt.draw()
    #-------------------
    
    
    #-------------------
    # Update Mask
    #-------------------
    def update_masked_figure(self, change):
        """
        Updates the figure based on the custom mask or mask parameters.
        """
        # Create the mask
        if self.custom_mask is not None:
            self.M.mask = self.custom_mask
        elif self.MDC.mask_yes_no_dropdown.value == 'Yes' and self.custom_mask is None:
            self.M.mask = self.M.create_mask(mask_xpos  = self.MDC.mask_xpos_text.value, 
                                             mask_ypos  = self.MDC.mask_ypos_text.value, 
                                             mask_shape = self.MDC.mask_shape_dropdown.value, 
                                             mask_size  = self.MDC.mask_size_text.value)
        else:
            self.M.mask = np.ones_like(self.ID.image_raw)

        # Mask the background colormap
        self.update_cmap()

        # Calculate the Last Closed Contour levels
        self.CL.calculate_lcc_levels()

        # Mask the new contours
        self.update_contours(self.M.mask)
    #-------------------
    
    
    #-------------------
    # Update CMap
    #-------------------
    def update_cmap(self):
        """
        Updates the background colormap based on interactive widget values.
        """
        self.ID.set_background_cmap(self.CL,
                                    self.M,
                                    selected_data=self.DC.dataSelector.value,
                                    selected_scaling=self.DC.scaling_toggle.value,
                                    cmapname=self.DC.colormap_dropdown.value,
                                    cmin=self.DC.clim_min_input.value,
                                    cmax=self.DC.clim_max_input.value)
        plt.draw()
    
    def on_clim_change(self, change):
        """
        Update the background colormap when the colarbar limits are changed
        """
        self.update_cmap()
    
    #-------------------
    
    
    #-------------------
    # Update Checkbox
    #-------------------
    def check_all(self, change, checkboxes_grid):
        """
        Checks all checkboxes.
        """
        import ipywidgets as widgets

        for i, checkbox in enumerate(checkboxes_grid.children):
            if isinstance(checkbox, widgets.Checkbox):
                checkbox.value = True
    
    def uncheck_all(self, change, checkboxes_grid):
        """
        Unchecks all checkboxes.
        """
        import ipywidgets as widgets

        for i, checkbox in enumerate(checkboxes_grid.children):
            if isinstance(checkbox, widgets.Checkbox):
                checkbox.value = False
    #-------------------
    
    
    #-------------------
    # Update Smoothing
    #-------------------
    def update_smoothing_parameters_from_widgets(self):
        """
        Updates smoothing parameters based on widget values.
        """
        for method, params in self.S.smoothing_parameters.items():
            for param_index, param_info in params.items():
                for widget_index, widget in enumerate(self.SC.smoothing_widgets[method]):
                    if param_info['label'] == widget.description:
                        param_info['value'] = widget.value

    def update_smoothing(self, change):
        """
        Smooths image according to selected parameters and updates the plot.
        """
        # Extract the smoothing method and update the associated smoothing parameters 
        selected_method = self.SC.smoothing_dropdown.value
        self.update_smoothing_parameters_from_widgets()

        # Smooth the image using the selected method and parameters 
        self.ID.image_smoothed = self.S.smooth_data(method=selected_method)

        # Update the background colormap
        self.update_cmap()

        # Calculate the new smoothed levels
        self.CL.calculate_smoothed_levels(self.AC.averaging_threshold_percentage.value)

        # Update the contours
        self.update_contours(self.M.mask, update_dataList_set=self.CL.dataList[self.CL.i_smoothed])
    
    # Update the layout to include the smoothing controls
    def update_smoothing_layout(self, change):
        """
        Updates the widget layout based on the selected smoothing method
        """
        selected_method = change.new
        new_controls = [self.DC.data_controls, self.CC.checkbox_controls, self.AC.averaging_controls, self.SC.smoothing_controls, self.RBC.remove_background_controls, self.MDC.mask_data_controls]
        self.SC.smoothing_controls.children = [self.SC.smoothing_dropdown, *self.SC.smoothing_widgets[selected_method]]
        self.main_layout.children = tuple(new_controls)
    #-------------------

    
    #-------------------
    # Update Remove Background
    #-------------------
    def update_nobackground_parameters_from_widgets(self):
        """
        Updates nobackground parameters based on widget values.
        """
        for param_index, widget in self.RBC.nobackground_widgets.items():
            self.NB.nobackground_parameters[param_index]['value'] = widget.value

    # Define an update function to recalculate smoothing parameters and update the plot
    def update_nobackground(self, change):
        """
        Removes image background according to selected parameters and updates the plot.
        """
        # Update the nobackground parameter dictionary with the new widget values
        self.update_nobackground_parameters_from_widgets()

        # Remove the background
        self.ID.image_nobackground, self.ID.estimated_background, self.ID.image_difference = self.NB.remove_background(self.ID.image_raw, self.RBC.filter_size_text.value, self.RBC.filter_shape_dropdown.value, self.RBC.pad_mode_dropdown.value)

        # Update the background colormap
        self.update_cmap()

        # Calculate new level values and update the contours
        self.CL.calculate_nobackground_levels(self.AC.averaging_threshold_percentage.value)
        self.update_contours(self.M.mask, update_dataList_set=self.CL.dataList[self.CL.i_nobackground])

    def update_nobackground_layout(self, change):
        """
        Updates the widget layout based on the selected nobackground method
        """
        selected_mode = self.RBC.nobackground_widgets[self.NB.i_pad_mode].value
        new_controls = [self.DC.data_controls, self.CC.checkbox_controls, self.AC.averaging_controls, self.SC.smoothing_controls, self.RBC.remove_background_controls, self.MDC.mask_data_controls]
        # If the selected mode is 'constant', include the pad_value widget
        if selected_mode == self.NB.padding_options[self.NB.i_constant]:
            self.RBC.remove_background_controls.children = [self.RBC.remove_background_label, self.RBC.filter_shape_dropdown, self.RBC.filter_size_text, self.RBC.pad_mode_dropdown, self.RBC.pad_value]
        else:
            self.RBC.remove_background_controls.children = [self.RBC.remove_background_label, self.RBC.filter_shape_dropdown, self.RBC.filter_size_text, self.RBC.pad_mode_dropdown]
        self.main_layout.children = tuple(new_controls)
    #-------------------
#-------------------End InteractivePlot


#***********************************************************
#===========================================================
#                 Core User Interface
#===========================================================
#***********************************************************


#-------------------
# Main User Interface
#-------------------
class UserInterface:
    """
    The ContourAnalysisTool provides users with a set of tools to analyse/process 2D images
    to interactively and programatically identify regions of interest (e.g. high density
    regions or background signals). The UserInterface class is a user-friendly interface
    that helps users analyse their image with minimal coding, either in its raw form or 
    after applying smoothing or background-removal techniques. These choices are referenced
    using the following keywords:

        raw:
            Applies thresholding directly to the original image without any preprocessing.
        
        smoothed:
            Smooths the image using a filter/kernel before applying thresholding, reducing 
            noise and enhancing the clarity of segmented regions.
        
        nobackground:
            Filters an image to remove background noise and/or global gradients before 
            applying thresholding.
    
    Techniques for determining regions of interest include searching for the last closed 
    contour, linear and logarithmic Otsu thresholding, and mean gradient thresholding.
    Keywords used for each of these techniques are as follows:

        LCC (Last Closed Contour):
            Identifies the last closed contour in the image (note: highly sensitive to 
            windowing/masking effects).
        
        Otsu:
            Automatically finds the best threshold value to distinguish between foreground 
            and background in an image by maximizing the difference in pixel intensities.
        
        OtsuLog:
            Applies a logarithmic transformation to the image before employing Otsu's 
            method (useful for low-contrast images).
        
        Average:
            Averages gradient magnitudes within a specified threshold to obtain a single 
            effective boundary level for an image, useful in identifying regions of interest.

    While the UserInterface is designed to be the main access point to the code, the
    underlying classes/functions/variables can still be directly manipulated/utilised by
    the user if desired. 

    The tool can be operated either interactively or programatically. The interactive feature 
    is particularly useful for quickly visualising changes to method parameters and/or 
    comparing different contour calculation methods and image processing techniques. The 
    simplest way to access the interactive tool is by running:

        ContourAnalysisTool.UserInterface(image, extent, interactive=True)

    where image is the user-provided data and extent = [xmin, xmax, ymin, ymax]. After the
    optimal configuration is obtained, the user can toggle the print button below the figure
    to obtain the numerical values of their selected contours and the executable code needed
    to reproduce those values and contours programatically.

    Alternatively, one can bypass the interactive feature and directly obtain the contour,
    the processed image data, and the figure. Here are some examples demonstrating how this
    can be done:

        CAT = ContourAnalysisTool.UserInterface(image, extent, mask=custom_mask, Nlevels=100, 
                                xy_units='pc', cbar_units='Jy', clabel='$\mathcal{F}$')
    
        levels_raw, image_raw, plot_raw = CAT.find_contours_raw(threshold=20, mask=CAT.M.create_mask())
        
        levels_smoothed, image_smoothed, plot_smoothed = CAT.find_contours_smoothed('Gaussian', 
            selected_scaling='Log', selected_levels=['Avg','otsulog','otsu','LCC'], sigma=1, threshold=90)
            
        levels_nobackground, image_nobackground, plot_nobackground = CAT.find_contours_nobackground(
            selected_levels=['lcc', 'otsu', 'average'], threshold=20, filter_shape='square')

    Since it can be difficult to remember all of the methods and parameters, there is
    some flexibility in the handling of string inputs and the code will attempt to match
    unknown inputs with available options. If the code is unable to find a match, it will
    resort to default values or throw an error with a list of available options. Also note
    that a mask can be input during the instantisation of the UserInterface or in the 
    individual find_contours functions. The former is typically used for custom masks while
    the later is used for simple masks created by the create_mask function in the Mask
    class, which first needs to be initialised before use.

    Dependencies include the following third-party Python packages:
        - numpy:
            Essential for numerical computations.
            (https://numpy.org/doc/stable/)
        
        - matplotlib:
            Comprehensive library for visualizations.
            (https://matplotlib.org/stable/contents.html)
        
        - scikit-image:
            Collection of image processing algorithms.
            (https://scikit-image.org/docs/stable/)
        
        - scipy:
            Library for scientific and technical computing.
            (https://docs.scipy.org/doc/scipy/reference/)
        
        - ipywidgets:
            Interactive HTML widgets for Jupyter notebooks.
            (https://ipywidgets.readthedocs.io/en/latest/)
        
        - IPython:
            Rich architecture for interactive computing.
            (https://ipython.readthedocs.io/en/stable/)
        
        - loess:
            LOESS (Local Regression) smoothing for 2D data.
            (https://pypi.org/project/loess/)
        
        - fuzzywuzzy:
            Library for fuzzy string matching.
            (https://github.com/seatgeek/fuzzywuzzy)
        
        - os:
            Operating system-dependent functionality.
    
    Missing packages can be installed using pip:
    pip install numpy matplotlib scikit-image scipy ipywidgets IPython loess fuzzywuzzy os
    """
    
    def __init__(self, image_raw, extent, interactive=False, mask=None, **kwargs):
        """
        Initialize the UserInterface object.

        Args:
            image_raw : 2D array-like
                The raw image provided by the user.
                
            extent : list of floats, [xmin, xmax, ymin, ymax]
                The x and y extent of the image.
                
            interactive : bool, optional, default: False
                Enables or disables interactive session.
                
            mask : 2D binary array-like, optional, default: None (reverts to np.ones_like(self.ID.image_raw))
                Custom mask to apply to the image.
                
            **kwargs : Additional optional keyword arguments (see self.ID.user_modifiable_plot_attributes).
                Nlevels : int, optional, default: 1000
                    The number of available contour levels in each image.
                    
                cmapname : str, optional, default: 'viridis'
                    The name of the color scheme used for the background colormap.
                    
                xy_units : str, optional, default: ''
                    The units of the x and y axes of the image.
                    
                cbar_units : str, optional, default: ''
                    The units of the color bar in the image.
                    
                xlabel : str, optional, default: 'x'
                    The label for the x-axis of the image plot.
                    
                ylabel : str, optional, default: 'y'
                    The label for the y-axis of the image plot.
                    
                clabel : str, optional, default: ''
                    The label for the color bar in the image plot.
                    
                line_style : list of str, optional, default: ['-', '--', ':']
                    The line style for each data type (Raw, Smoothed, NoBackground).
                    
                line_color : list of str or list of tuple, optional, default: ['red', 'bisque', 'lightsteelblue', 'orange']
                    The line color for each contour type (LCC, Otsu, OtsuLog, Average).
                    
                font_size : float, optional, default: 15
                    The font size of the text in the plot.
                    
                line_width : float, optional, default: 1
                    The line width of the contour lines.
        """
        # Initialise core modules
        self.ID  = ImageData(image_raw, extent)                
        self.S   = Smoothing(self.ID)
        self.NB  = NoBackground(self.ID)
        self.M   = Mask(self.ID)
        self.CL  = ContourLevels(self.ID, self.M)

        for key, value in kwargs.items():
            if hasattr(self.ID, key):
                setattr(self.ID, key, value)
            else:
                valid_attributes = self.ID.user_modifiable_plot_attributes
                raise AttributeError(f"No such attribute: {key}. "
                                     f"Valid attributes are: {valid_attributes}")

        self.set_mask(mask)

        if interactive:
            class_name = self.get_class_name()
            file_name  = self.get_current_filename()
            
            # Initialise all interactive modules
            self.MDC = MaskDataControls(self.ID, self.M)
            self.SC  = SmoothingControls(self.S)
            self.DC  = DataControls(self.ID, self.CL)
            self.CC  = CheckboxControls(self.CL)
            self.AC  = AveragingControls(self.CL)
            self.RBC = RemoveBackgroundControls(self.NB)
            self.PD  = PrintData(self.ID, self.M, self.CL, self.MDC, self.SC, self.DC, self.CC, self.AC, 
                                 class_name, file_name, self.get_smoothing_parameters, self.get_nobackground_params,
                                 custom_mask=mask)
            self.IP  = InteractivePlot(self.ID, self.S,  self.NB, self.M,   self.CL,  self.DC, 
                                       self.CC, self.AC, self.SC, self.RBC, self.MDC, self.PD, 
                                       custom_mask=mask)

    #-------------------
    # Raw
    #-------------------
    def find_contours_raw(self, selected_levels=None, selected_scaling=None, threshold=None, mask=None, display_plot=False):
        """
        Generate contours for raw image data.

        Args:
            selected_levels : list of str, optional, default: None (reverts to ['LCC', 'Otsu', 'OtsuLog', 'Average'])
                The levels to be used for contour generation.
                
            selected_scaling : str, optional, default: None (reverts to 'Linear')
                The scaling to be applied.
                
            threshold : float, optional, default: None (reverts to 20)
                The threshold value for contour calculation.
                
            mask : 2D array-like, optional, default: None (reverts to np.ones_like(self.ID.image_raw))
                The mask to be applied.
                
            display_plot : bool, optional, default: False
                Determines whether the figure will be output to the console or not.

        Returns:
            selected_levels_dict : dict
                Dictionary of selected contour levels.
                
            self.ID.image_raw : 2D array-like
                The raw image data.
                
            self.ID.hF : Figure
                Figure handle.
        """
        # Process the inputs
        selected_levels, selected_scaling, threshold = self.process_inputs(selected_levels, selected_scaling, threshold, mask)

        # Generate the contours
        selected_levels_dict = self.generate_contours(self.CL.i_raw, selected_levels, selected_scaling, threshold, display_plot)

        return selected_levels_dict, self.ID.image_raw, self.ID.hF
    #-------------------
    

    #-------------------
    # Smoothed
    #-------------------
    def find_contours_smoothed(self, selected_method, selected_levels=None, selected_scaling=None, threshold=None, mask=None, display_plot=False, **kwargs):
        """
        Generate contours for smoothed image data.

        Args:
            selected_method : str
                The smoothing method to be applied.
                
            selected_levels : list of str, optional, default: None (reverts to ['LCC', 'Otsu', 'OtsuLog', 'Average'])
                The levels to be used for contour generation.
                
            selected_scaling : str, optional, default: None (reverts to 'Linear')
                The scaling to be applied.
                
            threshold : float, optional, default: None (reverts to 20)
                The threshold value for contour calculation.
                
            mask : 2D array-like, optional, default: None (reverts to np.ones_like(self.ID.image_raw))
                The mask to be applied (default is None).
                
            display_plot : bool, optional, default: False
                Determines whether the figure will be output to the console or not.
                
            **kwargs: Additional parameters for smoothing.
                Gaussian:
                    sigma : float, optional, default: None (reverts to 2)
                        Standard deviation of the Gaussian kernel, controlling the extent of smoothing.
                
                Bivariate Spline:
                    s : float, optional, default: None (reverts to self.ID.rounded_min)
                        Controls the trade-off between fitting the data closely and producing a smooth surface.
        
                Local Regression (LOESS):
                    poly_order : int, optional, default: None (reverts to 2)
                        Degree of the polynomial to fit locally to the data.
                    locality_frac : float, optional, default: None (reverts to 0.01)
                        Proportion of data points considered for local regression.
        
                Savitzky-Golay:
                    window_length : int, optional, default: None (reverts to 10)
                        Number of data points used for polynomial fitting.
                    poly_order : int, optional, default: None (reverts to 2)
                        Degree of the polynomial to fit.
                    boundary_mode : str, optional, default: None (reverts to 'nearest')
                        Treatment of boundaries when applying the filter
                        Available options: ['nearest', 'constant', 'mirror', 'wrap', 'interp'].
        
                Moving Average:
                    kernel_radius : int, optional, default: None (reverts to 3)
                        Radius of the circular averaging kernel.
        
                Wiener Filtering:
                    mysize : int, optional, default: None (reverts to 5)
                        Size of the Wiener filter window.
                    noise : float, optional, default: None (reverts to 0.05)
                        Estimated noise standard deviation.
        
                Bilateral Filtering:
                    sigma_spatial : float, optional, default: None (reverts to 1)
                        Spatial Gaussian standard deviation.
                    sigma_range : float, optional, default: None (reverts to 1)
                        Intensity Gaussian standard deviation.
        
                Total Variation Denoising:
                    weight : float, optional, default: None (reverts to 0.5)
                        Weight parameter controlling the balance between data fidelity and smoothness.
                    n_iter : int, optional, default: None (reverts to 100)
                        Number of iterations for optimization.
        
                Anisotropic Diffusion:
                    n_iter : int, optional, default: None (reverts to 10)
                        Number of iterations for diffusion.
                    kappa : float, optional, default: None (reverts to 50)
                        Conduction coefficient controlling diffusion rate.
                    gamma : float, optional, default: None (reverts to 0.01)
                        Regularization parameter.
        
                Non-local Means Denoising:
                    patch_size : int, optional, default: None (reverts to 10)
                        Size of the patch used for similarity calculation.
                    h : float, optional, default: None (reverts to self.ID.rounded_min)
                        Filtering strength, controlling the degree of denoising.

        Returns:
            selected_levels_dict : dict
                Dictionary of selected contour levels.
                
            self.ID.image_smoothed : 2D array-like
                The smoothed image data.
                
            self.ID.hF : Figure
                Figure handle.
        """

        # Get default smoothing parameters
        selected_method, smoothing_params = self.get_smoothing_parameters(selected_method)
    
        # Update smoothing parameters with user-provided values
        for key, value in kwargs.items():
            if key in smoothing_params:
                smoothing_params[key] = value
            else:
                valid_params = list(smoothing_params.keys())
                raise ValueError(f"Invalid parameter '{key}' for selected method '{selected_method}'. "
                                 f"Valid options are: {valid_params}.")

        # Update the value in the smoothing parameter dictionary
        for param in self.S.smoothing_parameters[selected_method].values():
            if param['var_name'] in kwargs:
                param['value'] = kwargs[param['var_name']]

        # Process the inputs
        selected_levels, selected_scaling, threshold = self.process_inputs(selected_levels, selected_scaling, threshold, mask)

        # Smooth the image
        self.ID.image_smoothed = self.S.smooth_data(method=selected_method)

        # Generate the contours
        selected_levels_dict = self.generate_contours(self.CL.i_smoothed, selected_levels, selected_scaling, threshold, display_plot)

        return selected_levels_dict, self.ID.image_smoothed, self.ID.hF
        
    def get_smoothing_parameters(self, method):
        """
        Retrieve smoothing parameters for a given method.

        Args:
            method (str): The smoothing method.

        Returns:
            method: The validated smoothing method.
            smoothing_params: Dictionary of smoothing parameters.
        """
        if method not in self.S.smoothing_methods:
            method_index = self.expand_search(method, self.S.smoothing_methods)
            method = self.S.smoothing_methods[method_index]
    
        smoothing_params = {param['var_name']: param['value'] for param in self.S.smoothing_parameters[method].values()}
        return method, smoothing_params
    #-------------------

    
    #-------------------
    # No Background
    #-------------------
    def find_contours_nobackground(self, selected_levels=None, selected_scaling=None, threshold=None, mask=None, display_plot=False, **kwargs):
        """
        Generate contours for image data with background removed.

        Args:          
            selected_levels : list of str, optional, default: None (reverts to ['LCC', 'Otsu', 'OtsuLog', 'Average'])
                The levels to be used for contour generation.
                
            selected_scaling : str, optional, default: None (reverts to 'Linear')
                The scaling to be applied.
                
            threshold : float, optional, default: None (reverts to 20)
                The threshold value for contour calculation.
                
            mask : 2D array-like, optional, default: None (reverts to np.ones_like(self.ID.image_raw))
                The mask to be applied (default is None).
                
            display_plot : bool, optional, default: False
                Determines whether the figure will be output to the console or not.
                
            **kwargs: Additional parameters for background removal.
                filter_size : int, optional, default: None (reverts to 25)
                    Size of the filter used for background removal, affecting the size of the region considered for background estimation.
                    
                filter_shape : str, optional, default: None (reverts to 'disk')
                    Shape of the filter used for background removal, influencing the spatial extent of the background estimation.
                    Available options: ['disk', 'square']
                    
                pad_mode : str, optional, default: None (reverts to 'reflect')
                    Method for padding the image boundaries to reduce filter artifacts during background estimation.
                    Available options: ['reflect', 'symmetric', 'constant']
                
                pad_value : float, optional, default: None (reverts to 0)
                    Value used for padding the image when the padding mode is set to constant, controlling the value of the padding pixels.
            
        Returns:
            selected_levels_dict : dict
                Dictionary of selected contour levels.
                
            self.ID.image_nobackground : 2D array-like
                The no-background image data.
                
            self.ID.hF : Figure
                Figure handle.
        """
        # Get default nobackground parameters
        nobackground_params = self.get_nobackground_params()

        # Update nobackground parameters with user-provided values
        for key, value in kwargs.items():
            if key in nobackground_params:
                nobackground_params[key] = value
            else:
                valid_params = list(nobackground_params.keys())
                raise ValueError(f"Invalid parameter '{key}' for removing background. "
                                 f"Valid options are: {valid_params}.")

        # Update the value in the nobackground parameter dictionary
        for param_info in self.NB.nobackground_parameters.values():
            if param_info['var_name'] in kwargs:
                param_info['value'] = kwargs[param_info['var_name']]

        # Process the inputs
        selected_levels, selected_scaling, threshold = self.process_inputs(selected_levels, selected_scaling, threshold, mask)

        # Remove the background
        self.ID.image_nobackground, self.ID.estimated_background, self.ID.image_difference = self.NB.remove_background(self.ID.image_raw, **nobackground_params)

        # Generate the contours
        selected_levels_dict = self.generate_contours(self.CL.i_nobackground, selected_levels, selected_scaling, threshold, display_plot)

        return selected_levels_dict, self.ID.image_nobackground, self.ID.hF
        
    def get_nobackground_params(self):
        """
        Retrieve parameters for background removal.

        Returns:
            nobackground_params: Dictionary of background removal parameters.
        """
        nobackground_params = {}
        for param_index, param_info in self.NB.nobackground_parameters.items():
            nobackground_params[param_info['var_name']] = param_info['value']
        return nobackground_params
    #-------------------

    
    #-------------------
    # Process common inputs
    #-------------------
    def process_inputs(self, selected_levels, selected_scaling, threshold, mask):
        """
        Process common input parameters for contour generation.

        Args:
            selected_levels (list of str): The levels to be used for contour generation.
            selected_scaling (str): The scaling to be applied.
            threshold (float): The threshold value for contour calculation.
            mask (2D array-like): The mask to be applied.

        Returns:
            selected_levels: Processed contour levels.
            selected_scaling: Processed scaling option.
            threshold: Processed threshold value.
        """
        # Ensure that selected_levels is a list of strings
        if isinstance(selected_levels, str):
            selected_levels = [selected_levels]
            
        if threshold is None:
            threshold = self.CL.initial_averaging_threshold_percentage

        if selected_scaling is None:
            selected_scaling = ''

        self.set_mask(mask)

        return selected_levels, selected_scaling, threshold
    #-------------------

    
    #-------------------
    # Set the Mask
    #-------------------
    def set_mask(self, mask):
        """
        Set the mask for the image.

        Args:
            mask (2D array-like): The mask to be applied. If None, a default mask of ones is applied.
        """
        if mask is not None:
            self.M.mask = mask
        else:
            self.M.mask = np.ones_like(self.ID.image_raw)
    #-------------------


    #-------------------
    # Generate requested contours
    #-------------------
    def generate_contours(self, selected_data, selected_levels, selected_scaling, threshold, display_plot):
        """
        Generate contours for the selected data.

        Args:
            selected_data: A pointer to the data from which contours are to be generated (Raw, Smoothed, NoBackground).
            selected_levels (list of str): The levels to be used for contour generation (LCC, Otsu, OtsuLog, Average).
            selected_scaling (str): The scaling to be applied (Linear, Logarithmic).
            threshold (float): The threshold value used for the Average contour calculation.
            display_plot (bool): Determines whether the figure will be output to the console or not.

        Returns:
            selected_levels_dict: Dictionary of selected contour levels.
        """
        # Identify how the data should be scaled
        if selected_scaling not in (None, '') and selected_scaling not in self.ID.scaling_options:
            scaling_index    = self.expand_search(selected_scaling, self.ID.scaling_options)
            selected_scaling = self.ID.scaling_options[scaling_index]

        # Initialise the figure
        self.ID.initialise_figure(self.CL)

        # Set the background colormap
        self.ID.set_background_cmap(self.CL,
                                    self.M,
                                    selected_data=self.CL.dataList[selected_data],
                                    selected_scaling=selected_scaling,
                                    cmapname=self.ID.cmapname,
                                    cmin=self.ID.cbar_min,
                                    cmax=self.ID.cbar_max)

        # Calculate the selected contour levels
        if selected_data == self.CL.i_raw:
            self.CL.calculate_raw_levels(threshold)
        elif selected_data == self.CL.i_smoothed:
            self.CL.calculate_smoothed_levels(threshold)
        elif selected_data == self.CL.i_nobackground:
            self.CL.calculate_nobackground_levels(threshold)

        # Initialise the dictionaries
        self.CL.setup_contour_dictionaries()

        # Save the levels and setup a reduced dictionary of the selected levels
        levels = self.CL.contour_levels[self.CL.dataList[selected_data]]
        selected_levels_dict = {}
        if not selected_levels or selected_levels is None:
            selected_levels_dict = levels
        elif any(level not in levels for level in selected_levels):
            temp_dict = {}
            for level in selected_levels:
                if level in levels:
                    temp_dict[level] = levels[level]
                else:
                    # Assuming self.expand_search returns an index within bounds
                    level_index = self.expand_search(level, levels)
                    temp_dict[list(levels)[level_index]] = levels[list(levels)[level_index]]
            # Reorder to match original dictionaries
            selected_levels_dict = {key: temp_dict[key] for key in levels if key in temp_dict}
        else:
            for level in levels:
                if level in selected_levels:
                    selected_levels_dict[level] = levels[level]

        # Update selected_levels with the modified name list
        selected_levels = []
        for i, label in enumerate(selected_levels_dict):
            selected_levels.append(label)

        # Include the level spacing for each data object in the dictionary
        delta_label = 'delta_levels'
        if selected_data == self.CL.i_raw:
            selected_levels_dict[delta_label] = self.CL.delta_levels[self.CL.i_raw]
        elif selected_data == self.CL.i_smoothed:
            selected_levels_dict[delta_label] = self.CL.delta_levels[self.CL.i_smoothed]
        elif selected_data == self.CL.i_nobackground:
            selected_levels_dict[delta_label] = self.CL.delta_levels[self.CL.i_nobackground]

        # Make the contour objects
        for level in list(selected_levels_dict.keys()):
            if selected_data == self.CL.i_raw:
                if level == self.CL.levelList[self.CL.i_lcc]:
                    self.CL.make_lcc_contour()
                elif level == self.CL.levelList[self.CL.i_otsu]:
                    self.CL.make_otsu_contour()
                elif level == self.CL.levelList[self.CL.i_otsulog]:
                    self.CL.make_otsu_log_contour()
                elif level == self.CL.levelList[self.CL.i_average]:
                    self.CL.make_average_contour()
            elif selected_data == self.CL.i_smoothed:
                if level == self.CL.levelList[self.CL.i_lcc]:
                    self.CL.make_lcc_contour_smoothed()
                elif level == self.CL.levelList[self.CL.i_otsu]:
                    self.CL.make_otsu_contour_smoothed()
                elif level == self.CL.levelList[self.CL.i_otsulog]:
                    self.CL.make_otsu_log_contour_smoothed()
                elif level == self.CL.levelList[self.CL.i_average]:
                    self.CL.make_average_contour_smoothed()
            elif selected_data == self.CL.i_nobackground:
                if level == self.CL.levelList[self.CL.i_lcc]:
                    self.CL.make_lcc_contour_nobackground()
                elif level == self.CL.levelList[self.CL.i_otsu]:
                    self.CL.make_otsu_contour_nobackground()
                elif level == self.CL.levelList[self.CL.i_otsulog]:
                    self.CL.make_otsu_log_contour_nobackground()
                elif level == self.CL.levelList[self.CL.i_average]:
                    self.CL.make_average_contour_nobackground()

        # Update the contour dictionary with changed contours
        self.CL.setup_contour_dictionaries()

        # Update the contour lines
        self.CL.update_contour_lines(self.M.mask, 
                                     update_dataList_set=self.CL.dataList[selected_data], 
                                     update_levelList_set=list(selected_levels_dict.keys()))

        # Create a legend for the selected lines
        legend = self.create_legend(selected_data, selected_levels)

        # Close the figure to prevent it from displaying
        if not display_plot:
            plt.close(self.ID.hF)

        return selected_levels_dict
    #-------------------

    
    #-------------------
    # Correct for mispelled inputs
    #-------------------
    def expand_search(self, alias, option_list):
        """
        Correct for misspelled inputs using fuzzy matching.

        Args:
            alias (str): The input alias to search for.
            option_list (list of str): List of valid options.

        Returns:
            The index of the best match in the option list.
        """
        from fuzzywuzzy import fuzz
    
        # Extract method names from the option_list
        method_names = [entry.lower() if isinstance(entry, str) else entry['name'].lower() for entry in option_list]
    
        # Define method indices based on method names
        indices = {name: index for index, name in enumerate(method_names)}

        # Set variables to keep track of the closest match
        best_match_score = -1
        best_match_indices = []

        # Threshold value below which warnings are printed
        threshold = 40
    
        # Loop through each method and calculate the similarity score
        for name, index in indices.items():
            score = fuzz.token_sort_ratio(alias.lower(), name)
            if score > best_match_score:
                best_match_score = score
                best_match_indices = [index]
            elif score == best_match_score:
                best_match_indices.append(index)

        # Based on the calculated scores, decide on the best option and/or print some warnings
        scores = [fuzz.token_sort_ratio(alias.lower(), name) for name in indices.keys()]
        if best_match_score < threshold:
            warning = 'Warning! No matches found. Defaulting to:'
            potential_options = option_list
            result = 0
        else:
            warning = 'Warning! Multiple matches detected:'
            # Construct potential options
            potential_options = [(name, score) for name, score in zip(indices.keys(), scores) if score >= threshold]
            # Sort potential options by score in descending order
            potential_options.sort(key=lambda x: x[1], reverse=True)
            result = indices[potential_options[0][0]]  # Select the index of the option with the highest score
    
        # Print all potential options if no clear match
        if len(potential_options) > 1 and all(score != 100 for score in scores):
            default = ['==>','   ']
            print(warning)
            for index, name in enumerate(potential_options):
                if index == 0:
                    print(default[0], name)  # Print the first potential option with an arrow
                else:
                    print(default[1], name)  # Print the name of other potential options
    
        # Return the indices of the methods with the highest similarity score and the potential options
        return result
    #-------------------

    
    #-------------------
    # Create a Legend
    #-------------------
    def create_legend(self, selected_data, selected_levels):
        """
        Create a legend for the selected contours.

        Args:
            selected_data (int): The data for which the legend is created.
            selected_levels (list of str): The contour levels to be included in the legend.

        Returns:
            legend: The legend handle.
        """
        from matplotlib.lines  import Line2D

        # Initialize lists to store handles and labels for legend
        handles = []
        labels = []
    
        # Determine the corresponding contour dictionary and styles based on selected_data
        contour_dict = self.CL.contour_lines[self.CL.dataList[selected_data]]
        linestyle = self.ID.line_style[selected_data]
    
        # Iterate over contour labels to find the corresponding contour lines
        for i, label in enumerate(self.CL.contour_lines[self.CL.dataList[selected_data]]):
            if label in selected_levels:
                color = self.ID.line_color[i]
        
                # Create a Line2D object for the legend entry
                handles.append(Line2D([0], [0], linestyle=linestyle, linewidth=2, color=color))
        
                # Add the label (key) for the legend
                labels.append(label)
    
        # Create the legend with the handles and labels
        legend = self.ID.hA.legend(handles=handles, labels=labels, title=self.CL.dataList[selected_data], loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=len(handles), frameon=True, fancybox=True, shadow=True, fontsize=6)
    
        return legend
        #-------------------
    
    @classmethod
    def get_class_name(cls):
        """
        Retrieve the name of the class.

        Returns:
            The class name.
        """
        return cls.__name__

    
    @classmethod
    def get_current_filename(cls):
        """
        Get the filename of the file in which this class is defined.

        Returns:
            str: The filename of the current file without the extension.
        """
        import os

        # Get the file path of the current file
        file_path = os.path.abspath(__file__)
        
        # Extract the filename without extension
        filename = os.path.splitext(os.path.basename(file_path))[0]
        
        return filename
#-------------------

