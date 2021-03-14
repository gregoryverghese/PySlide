
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">PySlide</h3>
 
This is a library for preprocessing histological whole slide images (WSIs) and wraps around the OpenSlide package to extend the functionality on offer. The advent of Digital WSIs scanners have facilitated the use of computational methods in pathological research. WSIs have large storage requirements, due to their high resolution. This makes it difficult to apply numerically demanding algorithms at the image level.

Openslide is a C library that provides an easy way to interface with WSIs (other libraries that uncompress images into RAM cannot work with WSIs) and provides Python and Java bindingh. PySlide wraps around Openslide and will provide a richer set of functionality general preprocessing tasks needed before machine learning algorithms. At the core is the idea of a patching object that allows us to tile the WSI into a series of smaller patches. This is the beginnings of a comprehensive framework to work and manipulate WSIs particularly with a focus on machine learning. The aim is to provide a fullly equipt library such as ASAP that can save some of the unneccessary and tedious time creating boilerplace code to preprocess WSIs and is a useful tool for the digital pathology and medical AI community.

* a wrapper around openslide.OpenSlide class
* patching-based methods 
* generate mask representations
* measure class imbalance
* generate class weights
* sampling patchese
* save patches - HDF5,LMDB,tfrecords,PNG

### TODO

* add useful preprocesisng functions
* extend storage function
* need to add example with public dataset - BACH or Cameylon
* tests
* refactor code
* comment code

<!-- LICENSE -->
### License

Distributed under the MIT License

<!-- CONTACT -->
### Contact

gregory.verghese@gmail.com






