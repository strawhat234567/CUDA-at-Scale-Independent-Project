#include <iostream>
#include <string>
#include <vector>
#include <dirent.h> 
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp> 

// ---------------------------------------------------------
// Custom CUDA Kernel for Sobel Edge Detection
// ---------------------------------------------------------
__global__ void sobelEdgeDetectionKernel(const unsigned char* d_src, unsigned char* d_dst, int width, int height, size_t pitch) {
    // Calculate global thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure we stay within bounds and leave a 1-pixel border for the 3x3 filter
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Apply Sobel X and Y operators
        int Gx = -d_src[(y-1)*pitch + (x-1)] + d_src[(y-1)*pitch + (x+1)]
                 -2*d_src[y*pitch + (x-1)] + 2*d_src[y*pitch + (x+1)]
                 -d_src[(y+1)*pitch + (x-1)] + d_src[(y+1)*pitch + (x+1)];

        int Gy = -d_src[(y-1)*pitch + (x-1)] - 2*d_src[(y-1)*pitch + x] - d_src[(y-1)*pitch + (x+1)]
                 +d_src[(y+1)*pitch + (x-1)] + 2*d_src[(y+1)*pitch + x] + d_src[(y+1)*pitch + (x+1)];

        // Approximate magnitude
        int sum = abs(Gx) + abs(Gy); 
        if (sum > 255) sum = 255; // Clamp to max 8-bit value
        
        d_dst[y*pitch + x] = (unsigned char)sum;
    } else if (x < width && y < height) {
        // Set borders to black
        d_dst[y*pitch + x] = 0;
    }
}

// ---------------------------------------------------------
// Helper functions for Image I/O using OpenCV
// ---------------------------------------------------------
bool loadMonochromeImage(const std::string& filepath, unsigned char** host_data, int* width, int* height) {
    cv::Mat img = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Warning: Could not read image " << filepath << std::endl;
        return false;
    }
    *width = img.cols;
    *height = img.rows;
    
    size_t size = (*width) * (*height) * sizeof(unsigned char);
    *host_data = new unsigned char[size];
    std::memcpy(*host_data, img.data, size);
    return true;
}

void saveMonochromeImage(const std::string& filepath, unsigned char* host_data, int width, int height) {
    cv::Mat img(height, width, CV_8UC1, host_data);
    cv::imwrite(filepath, img);
}

// ---------------------------------------------------------
// Main CUDA execution
// ---------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " --input <input_dir> --output <output_dir>\n";
        return -1;
    }

    std::string input_dir = argv[2];
    std::string output_dir = argv[4];

    std::cout << "Starting Industrial Inspection Pipeline..." << std::endl;

    std::vector<std::string> files;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(input_dir.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if (filename.length() > 5 && filename.substr(filename.length() - 5) == ".tiff") {
                files.push_back(filename);
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error: Could not open input directory." << std::endl;
        return -1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int processed_count = 0;
    cudaEventRecord(start);

    for (const auto& file : files) {
        std::string input_path = input_dir + "/" + file;
        std::string output_path = output_dir + "/processed_" + file;

        int width = 0, height = 0;
        unsigned char* h_src = nullptr;

        if (!loadMonochromeImage(input_path, &h_src, &width, &height)) {
            continue;
        }

        // 1. Allocate Device Memory using cudaMallocPitch for proper memory alignment
        unsigned char *d_src = nullptr, *d_dst = nullptr;
        size_t pitch;
        cudaMallocPitch(&d_src, &pitch, width * sizeof(unsigned char), height);
        cudaMallocPitch(&d_dst, &pitch, width * sizeof(unsigned char), height);

        // 2. Copy data from Host to Device
        cudaMemcpy2D(d_src, pitch, h_src, width * sizeof(unsigned char), 
                     width * sizeof(unsigned char), height, cudaMemcpyHostToDevice);

        // 3. Define Block and Grid Dimensions
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // 4. Launch Custom Kernel
        sobelEdgeDetectionKernel<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, width, height, pitch);
        cudaDeviceSynchronize(); // Wait for GPU to finish

        // 5. Copy processed data back to Host
        unsigned char* h_dst = new unsigned char[width * height];
        cudaMemcpy2D(h_dst, width * sizeof(unsigned char), d_dst, pitch, 
                     width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);

        saveMonochromeImage(output_path, h_dst, width, height);
        processed_count++;

        // Free memory
        cudaFree(d_src);
        cudaFree(d_dst);
        delete[] h_src;
        delete[] h_dst;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "--------------------------------------------------\n";
    std::cout << "Inspection Complete!\n";
    std::cout << "Total Images Processed: " << processed_count << "\n";
    std::cout << "Total GPU Processing Time: " << milliseconds << " ms\n";
    if (processed_count > 0) {
        std::cout << "Average Time per Image: " << milliseconds / processed_count << " ms\n";
    }
    std::cout << "--------------------------------------------------\n";

    return 0;
}