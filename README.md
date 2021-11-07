
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="left">PySlide</h3>
 
A library for preprocessing histology slides (WSIs) with a focus for downstream machine learning tasks. Wraps around OpenSlide (a C library that provides an easy way to interface with WSIs) and extends the functionality to work with annotation files overlaid on the slides. This is the beginnings of a comprehensive framework to work and manipulate WSIs particularly with a focus on machine learning. Future versions will continue to build a useful tool for the digital pathology and medical AI community.

* generate annotation masks
* generate class labels
* extract regions of slide
* detect components on slide
* generate patches with different size, mag-level, step-size
* filter patches
* generate class weights
* stitch patches

### TODO

* sampling techniques
* preprocessing functions
* extend storage functionality (HDF5, LMDB, tfrecords)
* public datasets - BACH or Cameylon
* tests

<!-- LICENSE -->
### License

Distributed under the MIT License

<!-- CONTACT -->
### Contact

gregory.verghese@gmail.com






