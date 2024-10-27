
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="left">PySlide</h3>
 
PySlide is a comprehensive Python package for preprocessing pathology whole slide images (WSIs). Built as a wrapper around OpenSlide, it provides powerful, user-friendly functionality for working with high-resolution pathology images, making it ideal for researchers and data scientists in the medical imaging domain.

## Features

- **WSI Handling**: Supports large pathology slides and other WSI formats via OpenSlide.
- **Efficient Preprocessing**: Streamline tasks like cropping, resizing, and filtering at high performance.
- **Annotation Support**: Easily integrate and visualize annotations.
- **Tiling and Patching**: Flexible tiling options for patch extraction, ideal for deep learning workflows.
- **Image Metadata Extraction**: Retrieve and manage metadata from WSIs.
  
## Installation

Install PySlide via PyPI:

```bash
pip install PySlide
```

## Quick Start

```python
from pyslide import SlideProcessor

# Initialize the processor with your WSI file
processor = SlideProcessor("path/to/your/slide.svs")

# Example: Extract a tile
tile = processor.get_tile(x=1000, y=2000, width=256, height=256)

# Example: Preprocess a slide
processed_slide = processor.preprocess_slide(parameters)
```

<!-- LICENSE -->
### License

Distributed under the MIT License

<!-- CONTACT -->
### Contact

gregory.verghese@gmail.com






