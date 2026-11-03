#include "safetensors.h"

#include "json.h"

#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ggml.h"

namespace kt2 {

int64_t TensorView::elements() const {
    int64_t count = 1;
    for (int64_t d : shape) count *= d;
    return count;
}

float bf16ToFloat(uint16_t value) {
    uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

SafeTensors::SafeTensors(const std::string & path) : path_(path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("Could not open " + path);
    struct stat info {};
    if (::fstat(fd, &info) != 0 || info.st_size < 8) {
        ::close(fd);
        throw std::runtime_error(path + " is not a safetensors file");
    }
    length_ = static_cast<size_t>(info.st_size);
    base_ = ::mmap(nullptr, length_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (base_ == MAP_FAILED) {
        base_ = nullptr;
        throw std::runtime_error("Could not map " + path);
    }

    const auto * bytes = static_cast<const uint8_t *>(base_);
    uint64_t headerLength = 0;
    std::memcpy(&headerLength, bytes, sizeof(headerLength));
    if (headerLength > length_ - 8) throw std::runtime_error(path + " has a bad header");
    Json header = Json::parse(std::string(reinterpret_cast<const char *>(bytes + 8), headerLength));
    const uint8_t * data = bytes + 8 + headerLength;
    size_t dataLength = length_ - 8 - headerLength;

    for (const auto & [name, entry] : header.object) {
        if (name == "__metadata__") {
            for (const auto & [key, value] : entry.object) metadata_[key] = value.str();
            continue;
        }
        TensorView view;
        view.dtype = entry["dtype"].str();
        for (const auto & d : entry["shape"].array) view.shape.push_back(d.integer());
        auto begin = static_cast<size_t>(entry["data_offsets"][0].integer());
        auto end = static_cast<size_t>(entry["data_offsets"][1].integer());
        if (end < begin || end > dataLength) throw std::runtime_error(path + ": " + name + " lies outside the file");
        view.data = data + begin;
        view.bytes = end - begin;
        tensors_.emplace(name, std::move(view));
    }
}

SafeTensors::~SafeTensors() {
    if (base_) ::munmap(base_, length_);
}

const TensorView * SafeTensors::find(const std::string & name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const TensorView & SafeTensors::get(const std::string & name) const {
    const TensorView * found = find(name);
    if (!found) throw std::runtime_error(path_ + " has no tensor " + name);
    return *found;
}

std::vector<float> toFloat(const TensorView & tensor) {
    auto count = static_cast<size_t>(tensor.elements());
    std::vector<float> out(count);
    if (tensor.dtype == "F32") {
        std::memcpy(out.data(), tensor.data, count * sizeof(float));
    } else if (tensor.dtype == "F16") {
        ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(tensor.data), out.data(), static_cast<int64_t>(count));
    } else if (tensor.dtype == "BF16") {
        const auto * values = reinterpret_cast<const uint16_t *>(tensor.data);
        for (size_t i = 0; i < count; i++) out[i] = bf16ToFloat(values[i]);
    } else {
        throw std::runtime_error("Cannot read a " + tensor.dtype + " tensor as floats");
    }
    return out;
}

} // namespace kt2
