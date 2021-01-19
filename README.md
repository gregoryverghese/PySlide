<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!--
***[![Contributors][contributors-shield]][contributors-url]
***[![Forks][forks-shield]][forks-url]
***[![Stargazers][stars-shield]][stars-url]
***[![Issues][issues-shield]][issues-url]
***[![MIT License][license-shield]][license-url]
***[![LinkedIn][linkedin-shield]][linkedin-url]
-->


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
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>
    ·   
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Report Bug</a>
    ·   
    <a href="https://github.com/othneildrew/Best-README-Template/issues">Request Feature</a>
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
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This library provides functionality to preprocess histolpathological whole slide images. Digital whole slides are super high resolution images of a physical tissue specimen that has been removed during surgery for assessment by a pathologist. These specimens are stored on glass slides and scanned using a biological scanner to produce an image around the size of 100,000x100,000 pixels (1gb-4gb). 

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


<!-- GETTING STARTED -->
## Getting Started

How to start

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



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Gregory Verghese - [@your_twitter](https://twitter.com/your_username) - gregory.verghese@gmail.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
