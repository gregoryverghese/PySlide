
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">PySlide</h3>
 
    <br />
</a>
    <br />
    <br />
  </p>
</p>



This is a library for preprocessing histological whole slide images (WSIs) and wraps around the OpenSlide package to extend the functionality on offer. The advent of Digital WSIs scanners have facilitated the use of computational methods in pathological research but due to the high resolution nature of WSIs they often have large storage requirements which can be a burden in the application of numerically demanding algorithms.  

Openslide provides a nice framework to work with WSIs and provides a python API. This package will provide a richer set of functions on top of OpenSlide for general preprocessing. At the core we work by patching up the WSIs into smaller tiles. This is the beginnings of a comprehensive framework to work and manipulate WSIs particularly with a focus on machine learning.  

* a wrapper around openslide.OpenSlide class
* patching-based methods 
* generate mask representations
* measure class imbalance
* generate class weights
* sampling patchese
* save patches - HDF5,LMDB,tfrecords,PNG

I hope this library can save some of the unneccessary and tedious time creating boilerplace code to preprocess such WSIs and is a useful tool for the digital pathology and medical AI community.

### Built With

* [openslide](https://openslide.org/)
* [numpy](https://numpy.org/)
* [opencv](https://opencv.org/)

### Prerequisites

* openslide
  ```sh
  pip install openslide-python
  ```
* numpy
  ```sh
  pip install numpy 
  ```
* opencv
  ```sh
  pip install opencv-python
  ```

### Installation

Clone the repo
   ```sh
   git clone https://github.com/gregoryverghese/PySlide.git
   ```

### TODO

* add useful preprocesisng functions
* extend storage function
* need to add example with public dataset - BACH or Cameylon
* tests
* refactor code
* comment code

<!-- LICENSE -->
## License

Distributed under vi pre  the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

gregory.verghese@gmail.com






