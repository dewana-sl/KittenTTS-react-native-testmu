// A small JSON reader for safetensors headers and the check tool's reference
// files. Numbers are doubles; objects keep their keys in a std::map.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace kt2 {

struct Json {
    enum class Type { Null, Bool, Number, String, Array, Object };

    Type type = Type::Null;
    bool boolean = false;
    double number = 0;
    std::string string;
    std::vector<Json> array;
    std::map<std::string, Json> object;

    static Json parse(const std::string & text);

    /** The member `key`; throws when there is none. */
    const Json & operator[](const std::string & key) const;
    const Json & operator[](size_t index) const;
    /** The member `key`, or nullptr. */
    const Json * find(const std::string & key) const;
    size_t size() const { return type == Type::Array ? array.size() : object.size(); }

    double num() const;
    int64_t integer() const { return static_cast<int64_t>(num()); }
    const std::string & str() const;
};

} // namespace kt2
