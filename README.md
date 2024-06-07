<br/>
<div align="center">
<a href="https://github.com/markahutch/ContourAnalysisTool">
<img src="https://github.com/markahutch/ContourAnalysisTool/images/Logo.png" alt="Logo" width="100" height="100">
</a>
<h3 align="center">ContourAnalysisTool</h3>
<p align="center">
A Python toolset for analyzing and processing 2D contour maps to interactively and programmatically identify regions of interest.
<br/>
<br/>
<a href="https://github.com/markahutch/ContourAnalysisTool/"><strong>Explore the docs »</strong></a>
<br/>
<br/>
<a href="https://github.com/markahutch/ContourAnalysisTool/">View Demo .</a>  
<a href="https://github.com/markahutch/ContourAnalysisTool/issues/new?labels=bug&template=bug-report---.md">Report Bug .</a>
<a href="https://github.com/markahutch/ContourAnalysisTool/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
</p>
</div>

 ## About The Project

![Product Screenshot](https://source.unsplash.com/random/1920x1080)

The ContourAnalysisTool provides users with a set of tools to analyse/process 2D images to interactively and programatically identify regions of interest (e.g. high density regions or background signals).

 ### Dependencies

Importantly, interactive features have been exclusively developed/tested in a Jupyter environment using the ipywidgets package. Non-interactive features, on the other hand, should function properly in any Python environment. Here is a list of the third-party Python packages used by the code:

- [numpy](https://numpy.org/doc/stable/)
- [matplotlib](https://matplotlib.org/stable/contents.html)
- [scikit-image](https://scikit-image.org/docs/stable/)
- [scipy](https://docs.scipy.org/doc/scipy/reference/)
- [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/)
- [IPython](https://ipython.readthedocs.io/en/stable/)
- [loess](https://pypi.org/project/loess/)
- [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy)
    
Packages can be installed using pip in the usual manner. No other prerequisites are needed for the code.

 ## Getting Started

To start using the ContourAnalysisTool, simply download the source code (ContourAnalysisTool.py) and place the file in your current working directory. The package can then be imported in the usual manner:
    ```python
    import ContourAnalysisTool
    ```

Alternatively, you can clone the repo
    ```sh
    git clone https://github.com/markahutch/ContourAnalysisTool.git
    ```
and add the repo directory to your path prior to importing the package:
    ```python
    import sys
    sys.path.append('/path/to/repo/directory')

    import ContourAnalysisTool
    ```
   
 ## Usage

Once the package is imported, the UserInterface class provides a user-friendly interface that helps users analyse their image with minimal coding, either in its raw form or after applying smoothing or background-removal techniques. These choices are referenced using the following keywords:

- raw: Applies thresholding directly to the original image without any preprocessing.

- smoothed: Smooths the image using a filter/kernel before applying thresholding, reducing noise and enhancing the clarity of segmented regions.

- nobackground: Filters an image to remove background noise and/or global gradients before applying thresholding.
    
Techniques for determining regions of interest include searching for the last closed contour, linear and logarithmic Otsu thresholding, and mean gradient thresholding. Keywords used for each of these techniques are as follows:

- LCC (Last Closed Contour): Identifies the last closed contour in the image (note: highly sensitive to windowing/masking effects).

- Otsu: Automatically finds the best threshold value to distinguish between foreground and background in an image by maximizing the difference in pixel intensities.

- OtsuLog: Applies a logarithmic transformation to the image before employing Otsu's method (useful for low-contrast images).

- Average: Averages gradient magnitudes within a specified threshold to obtain a single effective boundary level for an image, useful in identifying regions of interest.

While the UserInterface is designed to be the main access point to the code, the
underlying classes/functions/variables can still be directly manipulated/utilised by
the user if desired.

The tool can be operated either interactively or programatically. The interactive feature  is particularly useful for quickly visualising changes to method parameters and/or comparing different contour calculation methods and image processing techniques. The simplest way to access the interactive tool is by running:
    ```python
    ContourAnalysisTool.UserInterface(image, extent, interactive=True)
    ```
where image is the user-provided data and extent = [xmin, xmax, ymin, ymax]. After the optimal configuration is obtained, the user can toggle the print button below the figure to obtain the numerical values of their selected contours and the executable code needed to reproduce those values and contours programatically.

Alternatively, one can bypass the interactive feature and directly obtain the contour, the processed image data, and the figure. Here are some examples demonstrating how this can be done:
    ```python
    CAT = ContourAnalysisTool.UserInterface(image, extent, mask=custom_mask      Nlevels=100, xy_units='pc', cbar_units='Jy', clabel='$\mathcal{F}$')      
    
    levels_raw, image_raw, plot_raw = CAT.find_contours_raw(threshold=20      mask=CAT.M.create_mask())
    
    levels_smoothed, image_smoothed, plot_smoothed = CAT.find_contours_smoothed('Gaussian', selected_scaling='Log', selected_levels=['Avg','otsulog','otsu','LCC'], sigma=1, threshold=90)
        
    levels_nobackground, image_nobackground, plot_nobackground = CAT.find_contours_nobackground(selected_levels=['lcc', 'otsu', 'average'], threshold=20, filter_shape='square')
    ```
For a list and description of the parameters available for the different functions, run one or more of the following commands:
    ```python
    help(CAT)
    help(CAT.find_contours_raw)
    help(CAT.find_contours_smoothed)
    help(CAT.find_contours_nobackground)
    ```

Since it can be difficult to remember all of the methods and parameters, there is some flexibility in the handling of string inputs and the code will attempt to match unknown inputs with available options. If the code is unable to find a match, it will resort to default values or throw an error with a list of available options. Also note that a mask can be input during the instantisation of the UserInterface or in the individual find_contours functions. The former is typically used for custom masks while the later is used for simple masks created by the create_mask function in the Mask class, which first needs to be initialised before use.


<!-- ## Roadmap-->
<!---->
<!--- [x] Add Changelog-->
<!--- [x] Add back to top links-->
<!--- [ ] Add Additional Templates w/ Examples-->
<!--- [ ] Add "components" document to easily copy & paste sections of the readme-->
<!--- [ ] Multi-language Support-->
<!--  - [ ] Chinese-->
<!--  - [ ] Spanish-->
<!---->
<!--See the [open issues](https://github.com/markahutch/ContourAnalysisTool/issues) for a full list of proposed features (and known issues).-->

 ## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

 ## License

Distributed under the GNU General Public License. See [GPL License](https://www.gnu.org/licenses/gpl-3.0.html) for more information.


<!-- ## Contact-->
<!---->
<!--Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com-->
<!---->
<!--Project Link: [https://github.com/markahutch/ContourAnalysisTool](https://github.com/markahutch/ContourAnalysisTool)-->
<!---->
<!-- ## Acknowledgments-->
<!---->
<!--Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!-->
