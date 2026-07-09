# CUDA Sobel Edge Detection Pipeline

This project implements a GPU-accelerated image processing pipeline for batch edge detection on grayscale TIFF images. It uses a custom CUDA kernel to apply the Sobel operator, processes images in parallel on the GPU, and writes the resulting edge-detected outputs to a specified output directory.

The program is designed to demonstrate CUDA-based parallel image processing, pitched memory allocation with `cudaMallocPitch`, and batch execution over a folder of input images.

## Features

- Custom CUDA `__global__` kernel for Sobel edge detection
- Batch processing of `.tiff` images from an input directory
- Grayscale image loading and saving with OpenCV
- Pitched GPU memory allocation with `cudaMallocPitch`
- Command-line arguments for input and output directories
- Execution timing with CUDA events

## Repository Contents

- `main.cu` — main CUDA source file containing image loading, GPU processing, and output saving
- `Makefile` — build instructions for compiling the CUDA program
- `README.md` — project description, setup, build, and usage instructions

## Requirements

Before building the project, make sure the following dependencies are installed:

- NVIDIA CUDA Toolkit
- OpenCV
- A C++ compiler compatible with CUDA
- Linux environment with directory access support

## Build Instructions

To compile the program, run:

```bash
make
```

If the build succeeds, the executable will be generated according to the rules defined in the `Makefile`.

## Usage

Run the program from the command line with an input directory and an output directory:

```bash
./main --input <input_directory> --output <output_directory>
```

Example:

```bash
./main --input data/input_textures --output data/output_textures
```

## How It Works

1. The program scans the input directory and collects all `.tiff` files.
2. Each image is loaded in grayscale using OpenCV.
3. The image is copied to GPU memory allocated with `cudaMallocPitch`.
4. A custom CUDA Sobel kernel computes edge magnitude for each pixel.
5. The processed image is copied back to host memory.
6. The output image is saved with a `processed_` filename prefix.
7. After all images are processed, the program prints total runtime and average time per image.

## Notes

- The program performs edge detection, not full defect classification.
- Output images highlight edges and structural changes that may support inspection tasks.
- The runtime reported by the program includes the batch-processing loop, including memory allocation, memory transfers, kernel execution, and synchronization.

## Style

This project is intended to follow the principles of the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html), including readable naming, clear comments, consistent formatting, and maintainable structure.

## Example Output

After execution, the program prints a summary similar to this:

```text
--------------------------------------------------
Inspection Complete!
Total Images Processed: 64
Total GPU Processing Time: 662.104 ms
Average Time per Image: 10.3454 ms
--------------------------------------------------
```

## Future Improvements

- Add CPU-versus-GPU benchmark comparison
- Add support for more input image formats
- Add error checking for CUDA API calls
- Add a post-processing stage for defect classification

## Author

Your Name Here