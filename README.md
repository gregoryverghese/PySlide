
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">PySlide</h3>
  
  <p align="center">UNDER DEVELOPMENT</p>

  <p align="center">
    An awesome histopathology whole slide image processing library!
    <br />
</a>
    <br />
    <br />
  </p>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>

  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This library provides functionality to preprocess histological whole slide images. Digital whole slides are super high resolution images of a physical tissue specimen that has been removed during surgery for assessment by a pathologist. These specimens are stored on glass slides and scanned using a biological scanner to produce an image around the size of 100,000x100,000 pixels (1gb-4gb). 

Due to their large memory footprint, analysing the images computationally can be a complex task, particularly in the contect of machine learning. Libraries such as openslide provide a nice framework to open and work with these images in languages like python but lack a richer set of functions for more advanced preprocessing amnd analysis. This library is the beginnings of a comprehensive framework to work and manipulate WSIs particularly with a focus on machine learning. For example, A number of approaches have been explored to ease the memory burden for training ml wholeslide image datasets including tiling the image into smaller more manageable patches. This library provides, a hopefully, simple way to perform such techniques and save down the resultant files. Functionality includes:

* a wrapper around openslides OpenSlide class to include annotations
* patching methods to split the image into smaller tiles
* generating mask representations of images for segmentation tasks
* measuring class imbalance and calculating weights
* sampling techniques
* saving images in a number of different formats - HDF5,LMDB,tfrecords,PNG

I hope this library can save some of the unneccessary and tedious time creating boilerplace code to preprocess such WSIs and is a useful tool for the digital pathology and medical AI community.

### Built With

* [openslide](https://openslide.org/)
* [numpy](https://numpy.org/)
* [opencv](https://opencv.org/)

### Prerequisites

Below are the installation steps for prerequities needed for PySlid

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

##Installation

Clone the repo
   ```sh
   git clone https://github.com/gregoryverghese/PySlide.git
   ```

<!-- USAGE EXAMPLES -->
## Usage

_For more examples, please refer to the [Documentation](https://example.com)_


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

gregory.verghese@gmail.com






