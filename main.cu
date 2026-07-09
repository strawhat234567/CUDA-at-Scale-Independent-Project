#include <iostream>
#include <string>
#include <vector>
#include <dirent.h> // For directory iteration
#include <cuda_runtime.h>
#include <nppi.h>   // NVIDIA Performance Primitives
#include <opencv2/opencv.hpp> // OpenCV core and I/O

// ---------------------------------------------------------
// Helper functions for Image I/O using OpenCV
// ---------------------------------------------------------
bool loadMonochromeImage(const std::string& filepath, unsigned char** host_data, int* width, int* height) {
    // Read the image in grayscale mode
    cv::Mat img = cv::imread(filepath, cv::IMREAD_GRAYSCALE);
    
    if (img.empty()) {
        std::cerr << "Warning: Could not read image " << filepath << std::endl;
        return false;
    }

    *width = img.cols;
    *height = img.rows;
    
    // Allocate host memory and copy the OpenCV pixel data over
    size_t size = (*width) * (*height) * sizeof(unsigned char);
    *host_data = new unsigned char[size];
    std::memcpy(*host_data, img.data, size);
    
    return true;
}

void saveMonochromeImage(const std::string& filepath, unsigned char* host_data, int width, int height) {
    // Wrap the raw host data back into an OpenCV matrix and save it
    cv::Mat img(height, width, CV_8UC1, host_data);
    cv::imwrite(filepath, img);
}

// ---------------------------------------------------------
// Main CUDA execution
// ---------------------------------------------------------
int main(int argc, char* argv[]) {
    // 1. CLI Argument Parsing
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " --input <input_dir> --output <output_dir>\n";
        return -1;
    }

    std::string input_dir = argv[2];
    std::string output_dir = argv[4];

    std::cout << "Starting Industrial Inspection Pipeline..." << std::endl;
    std::cout << "Input Directory: " << input_dir << std::endl;

    // 2. Read all .tiff files from the input directory
    std::vector<std::string> files;
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(input_dir.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            // Check if the file ends with .tiff
            if (filename.length() > 5 && filename.substr(filename.length() - 5) == ".tiff") {
                files.push_back(filename);
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error: Could not open input directory." << std::endl;
        return -1;
    }

    std::cout << "Found " << files.size() << " textures to process." << std::endl;

    // 3. Setup CUDA Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int processed_count = 0;
    cudaEventRecord(start);

    // 4. Batch Processing Loop
    for (const auto& file : files) {
        std::string input_path = input_dir + "/" + file;
        std::string output_path = output_dir + "/processed_" + file;

        int width = 0, height = 0;
        unsigned char* h_src = nullptr;

        if (!loadMonochromeImage(input_path, &h_src, &width, &height)) {
            continue;
        }

        // Allocate Device (GPU) Memory using NPP's optimized allocator
        int src_step, dst_step;
        Npp8u* d_src = nppiMalloc_8u_C1(width, height, &src_step);
        Npp8u* d_dst = nppiMalloc_8u_C1(width, height, &dst_step);

        // Copy data from Host to Device
        cudaMemcpy2D(d_src, src_step, h_src, width * sizeof(unsigned char), 
                     width * sizeof(unsigned char), height, cudaMemcpyHostToDevice);

        // Define Region of Interest (ROI) - Process the whole image
        NppiSize roiSize = {width, height};

        // Execute NPP Filter (Sobel Edge Detection for 8-bit, 1-channel images)
        nppiFilterSobel_8u_C1R(d_src, src_step, d_dst, dst_step, roiSize, NPP_MASK_SIZE_3_X_3);

        // Copy processed data back to Host
        unsigned char* h_dst = new unsigned char[width * height];
        cudaMemcpy2D(h_dst, width * sizeof(unsigned char), d_dst, dst_step, 
                     width * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);

        // Save the output
        saveMonochromeImage(output_path, h_dst, width, height);
        processed_count++;

        // Free memory for this iteration
        nppiFree(d_src);
        nppiFree(d_dst);
        delete[] h_src;
        delete[] h_dst;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 5. Generate Proof of Execution Metrics
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