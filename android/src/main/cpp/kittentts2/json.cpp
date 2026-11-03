#include "json.h"

#include <cstdlib>
#include <stdexcept>

namespace kt2 {

namespace {

struct Parser {
    const std::string & text;
    size_t at = 0;

    [[noreturn]] void fail(const char * what) const {
        throw std::runtime_error(std::string("JSON: ") + what + " at offset " + std::to_string(at));
    }

    void space() {
        while (at < text.size() && (text[at] == ' ' || text[at] == '\n' || text[at] == '\r' || text[at] == '\t')) at++;
    }

    bool consume(char c) {
        space();
        if (at < text.size() && text[at] == c) { at++; return true; }
        return false;
    }

    void expect(char c) {
        if (!consume(c)) fail("unexpected character");
    }

    static void appendUtf8(std::string & out, uint32_t code) {
        if (code < 0x80) {
            out += static_cast<char>(code);
        } else if (code < 0x800) {
            out += static_cast<char>(0xC0 | (code >> 6));
            out += static_cast<char>(0x80 | (code & 0x3F));
        } else if (code < 0x10000) {
            out += static_cast<char>(0xE0 | (code >> 12));
            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code & 0x3F));
        } else {
            out += static_cast<char>(0xF0 | (code >> 18));
            out += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
            out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (code & 0x3F));
        }
    }

    uint32_t hex4() {
        if (at + 4 > text.size()) fail("short \\u escape");
        uint32_t value = 0;
        for (int i = 0; i < 4; i++) {
            char c = text[at++];
            value <<= 4;
            if (c >= '0' && c <= '9') value |= c - '0';
            else if (c >= 'a' && c <= 'f') value |= c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') value |= c - 'A' + 10;
            else fail("bad \\u escape");
        }
        return value;
    }

    std::string stringLiteral() {
        expect('"');
        std::string out;
        while (true) {
            if (at >= text.size()) fail("unterminated string");
            char c = text[at++];
            if (c == '"') return out;
            if (c != '\\') { out += c; continue; }
            if (at >= text.size()) fail("unterminated escape");
            char e = text[at++];
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t code = hex4();
                    if (code >= 0xD800 && code < 0xDC00 && at + 6 <= text.size() && text[at] == '\\' && text[at + 1] == 'u') {
                        at += 2;
                        uint32_t low = hex4();
                        code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                    }
                    appendUtf8(out, code);
                    break;
                }
                default: fail("bad escape");
            }
        }
    }

    Json value() {
        space();
        if (at >= text.size()) fail("unexpected end");
        Json out;
        char c = text[at];
        if (c == '{') {
            at++;
            out.type = Json::Type::Object;
            if (consume('}')) return out;
            do {
                space();
                std::string key = stringLiteral();
                expect(':');
                out.object.emplace(std::move(key), value());
            } while (consume(','));
            expect('}');
        } else if (c == '[') {
            at++;
            out.type = Json::Type::Array;
            if (consume(']')) return out;
            do {
                out.array.push_back(value());
            } while (consume(','));
            expect(']');
        } else if (c == '"') {
            out.type = Json::Type::String;
            out.string = stringLiteral();
        } else if (text.compare(at, 4, "true") == 0) {
            at += 4;
            out.type = Json::Type::Bool;
            out.boolean = true;
        } else if (text.compare(at, 5, "false") == 0) {
            at += 5;
            out.type = Json::Type::Bool;
        } else if (text.compare(at, 4, "null") == 0) {
            at += 4;
        } else {
            const char * start = text.c_str() + at;
            char * end = nullptr;
            out.number = std::strtod(start, &end);
            if (end == start) fail("bad value");
            out.type = Json::Type::Number;
            at += static_cast<size_t>(end - start);
        }
        return out;
    }
};

} // namespace

Json Json::parse(const std::string & text) {
    Parser parser{text};
    Json out = parser.value();
    parser.space();
    if (parser.at != text.size()) parser.fail("trailing characters");
    return out;
}

const Json & Json::operator[](const std::string & key) const {
    const Json * found = find(key);
    if (!found) throw std::runtime_error("JSON: no member " + key);
    return *found;
}

const Json & Json::operator[](size_t index) const {
    if (type != Type::Array || index >= array.size()) throw std::runtime_error("JSON: index out of range");
    return array[index];
}

const Json * Json::find(const std::string & key) const {
    if (type != Type::Object) return nullptr;
    auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
}

double Json::num() const {
    if (type != Type::Number) throw std::runtime_error("JSON: not a number");
    return number;
}

const std::string & Json::str() const {
    if (type != Type::String) throw std::runtime_error("JSON: not a string");
    return string;
}

} // namespace kt2
