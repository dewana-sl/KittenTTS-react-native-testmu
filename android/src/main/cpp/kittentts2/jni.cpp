// JNI bridge between KittenTTS2GGMLModule.java and the ggml engine (engine.h).
// Errors become RuntimeExceptions; a cancelled generation becomes a
// CancellationException.

#include <jni.h>

#include <android/log.h>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#include "engine.h"

using kt2::Engine;

namespace {

const char * kTag = "KittenTTS2";

/** ggml's own messages (ggml-vulkan's device report, its errors) go to logcat. */
void logGgmlToLogcat() {
    static const bool installed = [] {
        ggml_log_set([](ggml_log_level level, const char * text, void *) {
            const int priority = level == GGML_LOG_LEVEL_ERROR ? ANDROID_LOG_ERROR
                               : level == GGML_LOG_LEVEL_WARN  ? ANDROID_LOG_WARN
                               : level == GGML_LOG_LEVEL_DEBUG ? ANDROID_LOG_DEBUG
                                                               : ANDROID_LOG_INFO;
            __android_log_print(priority, "ggml", "%s", text);
        }, nullptr);
        return true;
    }();
    (void) installed;
}

std::vector<int> ints(JNIEnv * env, jintArray array) {
    std::vector<int> out(static_cast<size_t>(env->GetArrayLength(array)));
    env->GetIntArrayRegion(array, 0, static_cast<jsize>(out.size()), reinterpret_cast<jint *>(out.data()));
    return out;
}

std::vector<float> floats(JNIEnv * env, jfloatArray array) {
    std::vector<float> out(static_cast<size_t>(env->GetArrayLength(array)));
    env->GetFloatArrayRegion(array, 0, static_cast<jsize>(out.size()), out.data());
    return out;
}

std::string text(JNIEnv * env, jstring value) {
    const char * chars = env->GetStringUTFChars(value, nullptr);
    std::string out(chars);
    env->ReleaseStringUTFChars(value, chars);
    return out;
}

void rethrow(JNIEnv * env) {
    try {
        throw;
    } catch (const kt2::Cancelled & error) {
        env->ThrowNew(env->FindClass("java/util/concurrent/CancellationException"), error.what());
    } catch (const std::exception & error) {
        __android_log_print(ANDROID_LOG_ERROR, kTag, "%s", error.what());
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), error.what());
    } catch (...) {
        env->ThrowNew(env->FindClass("java/lang/RuntimeException"), "Unknown error in the ggml engine");
    }
}

double since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

} // namespace

extern "C" {

#define KT2_JNI(name) JNIEXPORT JNICALL Java_com_kittentts_reactnative_KittenTTS2GGMLModule_##name

/** Whether this CPU has the instructions ggml was built for (dot product and fp16 on arm64). */
jboolean KT2_JNI(nativeSupported)(JNIEnv *, jclass) {
#if defined(__aarch64__)
    const unsigned long caps = getauxval(AT_HWCAP);
    return (caps & HWCAP_ASIMDDP) && (caps & HWCAP_FPHP) && (caps & HWCAP_ASIMDHP) ? JNI_TRUE : JNI_FALSE;
#else
    return JNI_TRUE;
#endif
}

jstring KT2_JNI(nativeDevice)(JNIEnv * env, jclass) {
    try {
        logGgmlToLogcat();
        const std::string device = Engine::shared().device();
        __android_log_print(ANDROID_LOG_INFO, kTag, "runs on %s", device.c_str());
        return env->NewStringUTF(device.c_str());
    } catch (...) {
        rethrow(env);
        return nullptr;
    }
}

void KT2_JNI(nativeLoad)(JNIEnv * env, jclass, jstring path, jintArray config, jfloatArray floatConfig) {
    try {
        const std::vector<int> c = ints(env, config);
        const std::vector<float> f = floats(env, floatConfig);
        if (c.size() != 6 || f.size() != 2) throw std::runtime_error("Bad language model config");
        kt2::LanguageModelConfig lm;
        lm.hiddenSize = c[0];
        lm.layers = c[1];
        lm.heads = c[2];
        lm.kvHeads = c[3];
        lm.headDim = c[4];
        lm.headStart = c[5];
        lm.ropeTheta = f[0];
        lm.rmsNormEps = f[1];
        logGgmlToLogcat();
        const auto start = std::chrono::steady_clock::now();
        Engine::shared().load(text(env, path), lm);
        __android_log_print(ANDROID_LOG_INFO, kTag, "language model loaded on %s in %.2fs",
                            Engine::shared().device().c_str(), since(start));
    } catch (...) {
        rethrow(env);
    }
}

void KT2_JNI(nativeLoadDecoder)(JNIEnv * env, jclass, jstring path) {
    try {
        const auto start = std::chrono::steady_clock::now();
        Engine::shared().loadDecoder(text(env, path));
        __android_log_print(ANDROID_LOG_INFO, kTag, "audio decoder loaded in %.2fs", since(start));
    } catch (...) {
        rethrow(env);
    }
}

/**
 * settings: temperature, topK, topP, minP, repetitionPenalty, repetitionWindow
 * (NaN for none), tokenRunPenalty, tokenRunGrace. ids: speechEnd, stop, audioStart, audioEnd.
 */
jintArray KT2_JNI(nativeGenerate)(JNIEnv * env, jclass, jintArray prompt, jfloatArray speaker, jfloatArray settings,
                                  jintArray ids, jint maxNewTokens, jint seed) {
    try {
        const std::vector<float> s = floats(env, settings);
        const std::vector<int> id = ints(env, ids);
        if (s.size() != 8 || id.size() != 4) throw std::runtime_error("Bad sampling settings");
        kt2::GenerateOptions options;
        options.sampling.temperature = s[0];
        options.sampling.topK = static_cast<int>(s[1]);
        options.sampling.topP = s[2];
        options.sampling.minP = s[3];
        options.sampling.repetitionPenalty = s[4];
        if (!std::isnan(s[5])) options.sampling.repetitionWindow = static_cast<int>(s[5]);
        options.sampling.tokenRunPenalty = s[6];
        options.sampling.tokenRunGrace = static_cast<int>(s[7]);
        options.speechEnd = id[0];
        options.stop = id[1];
        options.audioStart = id[2];
        options.audioEnd = id[3];
        options.maxNewTokens = maxNewTokens;
        options.seed = static_cast<uint32_t>(seed);
        const std::vector<int> tokens = Engine::shared().generate(ints(env, prompt), floats(env, speaker), options);
        jintArray out = env->NewIntArray(static_cast<jsize>(tokens.size()));
        env->SetIntArrayRegion(out, 0, static_cast<jsize>(tokens.size()), reinterpret_cast<const jint *>(tokens.data()));
        return out;
    } catch (...) {
        rethrow(env);
        return nullptr;
    }
}

/** float32 samples as little-endian bytes. */
jbyteArray KT2_JNI(nativeDecode)(JNIEnv * env, jclass, jintArray tokens, jintArray promptToken, jfloatArray promptFeat,
                                 jfloatArray embedding, jint seed) {
    try {
        const std::vector<float> samples = Engine::shared().decode(
            ints(env, tokens), ints(env, promptToken), floats(env, promptFeat), floats(env, embedding),
            static_cast<uint32_t>(seed));
        const auto bytes = static_cast<jsize>(samples.size() * sizeof(float));
        jbyteArray out = env->NewByteArray(bytes);
        env->SetByteArrayRegion(out, 0, bytes, reinterpret_cast<const jbyte *>(samples.data()));
        return out;
    } catch (...) {
        rethrow(env);
        return nullptr;
    }
}

void KT2_JNI(nativeCancel)(JNIEnv *, jclass) {
    Engine::shared().cancel();
}

void KT2_JNI(nativeUnload)(JNIEnv * env, jclass) {
    try {
        Engine::shared().unload();
    } catch (...) {
        rethrow(env);
    }
}

} // extern "C"
