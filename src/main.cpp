#include <cuda_runtime.h>
// Allows us to do big matrix multiplications.
#include <cublas_v2.h>
#include <cstdlib>
#include <queue>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
// Grabs json header for functions to parse and create json objects in cpp.
#include <nlohmann/json.hpp>
#include <fstream>
#include <numeric>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <array>
#include <vector>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <limits>
#include "tokenizer.hpp"

using json = nlohmann::json;

#include <iostream>

constexpr size_t B_TO_MB = 1024 * 1024;
constexpr size_t B_TO_GB = 1024 * 1024 * 1024;
using json = nlohmann::json;

int checkGPUStatus(){
    int device_count = 0;
    cudaGetDeviceCount (&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found\n");
        return -1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem / B_TO_MB << " MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem / B_TO_GB << "GB, total memory: " << total_mem / B_TO_GB << "GB\n";
    return 0;
}

struct LLamamWeights{
    void *model_stoage = nullptr;
    __nv_bfloat16 *embed_tokens = nullptr;
    __nv_bfloat16 *lm_head = nullptr;
    __nv_bfloat16 *norm = nullptr;

    __nv_bfloat16 *norm = nullptr;
    __nv_bfloat16 *input_laternorm[n_LAYERS]{};
    __nv_bfloat16 *post_attn_layernorms[N_LAYERS]{};

    __nv_bfloat16 *w_q[N_LAYERS]{};
    __nv_bfloat16 *w_k[N_LAYERS]{};
    __nv_bfloat16 *w_v[N_LAYERS]{};
    __nv_bfloat16 *w_o[N_LAYERS]{};

    __nv_bfloat16 *mlp_gate_proj[N_LAYERS]{};
    __nv_bfloat16 *mlp_up_proj[N_LAYERS]{};
    __nv_bfloat16 *mlp_down_proj[N_LAYERS]{};

};

int loadLamaModel(LLamaWeights &wweights){
    std::string path = "models/llama-3.2-1b-instruct/model.safetensors";
    //open file as binary file
    std::ifstream safetensors_file(path, std::ios::binary);
    if (!safetensors_file.is_open()) {
        std::cerr << "Failed to open safetensors file: " << path << std::endl;
        return -1;
    }
    // Read the header size (8 bytes) from the file
    uint64_t headersize = 0;
    safetensors_file.read(reinterpret_cast<char*>(&headersize), 8);
    if(!safetensors_file) {
        std::cerr << "Failed to read header size from safetensors file." << std::endl;
        return -1;
    }
    std::string header(headersize, '\0');
    // header.data points to string's actual character buffer.
    safetensors_file.read(header.data(), headersize);

    // Convert this bigass string into a json object.
    json header_json = json::parse(header);

    //create a hashmap to store the tensor metadata
    std::unordered_map<std::string, uint64_t> offset;
    uint64_t max_offset = 0;
    
    //read-only access jey and value pairs in the json header
    for (const auto &[name , tensor_info]: header_json.items())
    {
        if (name == "__metadata__") {
            continue; // Skip the metadata entry
        }
        const auto &data_offsets = tensor_info.at("data_offsets");

        uint64_t start_offset = data_offsets.at(0).get<uint64_t>();
        uint64_t end_offset = data_offsets.at(1).get<uint64_t>();

        offsets.emplace(name, start_offset);
        max_offset = std::max(max_offset, end_offset);
    }
    std::vector<char> model_weights_cpu(max_offset);
    safetensors_file.read(model_weights_cpu.data(), 
    static_cast<std::streamsize>(model_weights_cpu.size()));

    // Copy the model weights from CPU to GPU
    void *model_weight_gpu = nullptr;
    cudaMalloc(&model_weight_gpu , max_offset);

    cudaMemcpy(model_weight_gpu, model_weights_cpu.data(), model_weights_cpu.size(), cudaMemcpyHostToDevice);

    weights.model_storage = model_weights_gpu;
    char *base = static_cast<char *>(weights.model_storage);

    weights.embed_tokens = reinterpret_cast<__nv_bfloat16 *>(
        base + offsets.at("model.embed_tokens.weight"));
    weights.lm_head = offsets.count("lm_head.weight") != 0
                          ? reinterpret_cast<__nv_bfloat16 *>(base + offsets.at("lm_head.weight"))
                          : weights.embed_tokens;

    weights.norm = reinterpret_cast<__nv_bfloat16 *>(base + offsets.at("model.norm.weight"));

    for (int layer = 0; layer< N_LAYERS ; ++layer){
        std::string prefix = "model.layers." + std::to_string(layer);
        weights.w_q[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".self_attn.q_proj.weight"));
        weights.w_k[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".self_attn.k_proj.weight"));
        weights.w_v[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".self_attn.v_proj.weight"));
        weights.w_o[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".self_attn.o_proj.weight"));

        weights.input_layernorm[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".input_layernorm.weight"));

        weights.post_attn_layernorms[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".post_attention_layernorm.weight"));

        weights.mlp_gate_proj[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".mlp.gate_proj.weight"));

        weights.mlp_up_proj[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".mlp.up_proj.weight"));

        weights.mlp_down_proj[layer] = reinterpret_cast<__nv_bfloat16 *>(
            base + offsets.at(prefix + ".mlp.down_proj.weight"));
        
    }
    return 0;
}