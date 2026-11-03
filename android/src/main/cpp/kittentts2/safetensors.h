// Read-only, memory-mapped safetensors files: the tensors stay in the page cache
// and are read where they lie.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace kt2 {

struct TensorView {
    std::string dtype;            // "F16", "BF16", "F32", "U8", "I32", ...
    std::vector<int64_t> shape;   // outermost first, as torch writes it
    const uint8_t * data = nullptr;
    size_t bytes = 0;

    int64_t elements() const;
};

class SafeTensors {
public:
    explicit SafeTensors(const std::string & path);
    ~SafeTensors();
    SafeTensors(const SafeTensors &) = delete;
    SafeTensors & operator=(const SafeTensors &) = delete;

    const TensorView * find(const std::string & name) const;
    /** The tensor `name`; throws when there is none. */
    const TensorView & get(const std::string & name) const;
    const std::map<std::string, TensorView> & tensors() const { return tensors_; }
    const std::map<std::string, std::string> & metadata() const { return metadata_; }

private:
    std::string path_;
    void * base_ = nullptr;
    size_t length_ = 0;
    std::map<std::string, TensorView> tensors_;
    std::map<std::string, std::string> metadata_;
};

/** float32 from a tensor of F32, F16 or BF16 values. */
std::vector<float> toFloat(const TensorView & tensor);

float bf16ToFloat(uint16_t value);

} // namespace kt2
