
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">PySlide</h3>
 
This is a library for preprocessing histology slides (WSIs) with a focus for downstream machine learning tasks. It wraps around OpenSlide (a C library that provides an easy way to interface with WSIs) and extends the library functionality to work with annotation files overlaid on the slides. This is the beginnings of a comprehensive framework to work and manipulate WSIs particularly with a focus on machine learning. Future versions will continue to extend the functinality and build a useful tool for the digital pathology and medical AI community.

* generate annotation mask
* extract regions of slide
* generate patches at all mag levels
* generate class weights
* stitch patches

### TODO

* sampling technique
* add useful preprocesisng functions
* extend storage function (HDF5, LMDB, tfrecords)
* public dataset - BACH or Cameylon
* tests

<!-- LICENSE -->
### License

Distributed under the MIT License

<!-- CONTACT -->
### Contact

gregory.verghese@gmail.com






