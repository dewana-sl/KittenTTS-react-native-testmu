package com.kittentts.reactnative;

import android.util.Base64;
import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.WritableArray;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * React Native's view of Kitten TTS 2's ggml engine (src/main/cpp/kittentts2):
 * the language model and the audio decoder on the GPU through Vulkan, or on the
 * CPU. It has the calls of the iOS MLX module (ios/KittenTTS2MLXModule.m), and
 * runs them one at a time on its own thread. libkittentts2.so is only in apps
 * built with kittentts.ggml=true; without it the module reports itself unavailable.
 */
public final class KittenTTS2GGMLModule extends ReactContextBaseJavaModule {
  private static final boolean LOADED = loadLibrary();

  private final ExecutorService queue = Executors.newSingleThreadExecutor(runnable -> {
    Thread thread = new Thread(runnable, "kittentts2-ggml");
    thread.setDaemon(true);
    return thread;
  });

  public KittenTTS2GGMLModule(ReactApplicationContext context) {
    super(context);
  }

  private static boolean loadLibrary() {
    try {
      System.loadLibrary("kittentts2");
      return true;
    } catch (Throwable error) {
      // Not built, a 32-bit ABI, or Android below 9 (no Vulkan 1.1 entry points).
      return false;
    }
  }

  @Override
  public String getName() {
    return "KittenTTS2GGML";
  }

  @ReactMethod
  public void isAvailable(Promise promise) {
    promise.resolve(LOADED && nativeSupported());
  }

  /** "Vulkan0: <GPU>", or "CPU (<why the GPU is not used>)". The first call looks for the GPU. */
  @ReactMethod
  public void device(Promise promise) {
    run(promise, "GGML_DEVICE_FAILED", () -> nativeDevice());
  }

  @ReactMethod
  public void load(String path, ReadableMap config, Promise promise) {
    run(promise, "GGML_LOAD_FAILED", () -> {
      int[] ints = {
        config.getInt("hidden_size"),
        config.getInt("num_hidden_layers"),
        config.getInt("num_attention_heads"),
        config.getInt("num_key_value_heads"),
        config.getInt("head_dim"),
        config.getInt("head_start"),
      };
      float[] floats = {
        config.hasKey("rope_theta") ? (float) config.getDouble("rope_theta") : 1e6f,
        config.hasKey("rms_norm_eps") ? (float) config.getDouble("rms_norm_eps") : 1e-6f,
      };
      nativeLoad(path, ints, floats);
      return null;
    });
  }

  /** Returns the token ids written after the prompt, up to `<speech_end>` or `[STOP]`. */
  @ReactMethod
  public void generate(ReadableArray prompt, ReadableArray speaker, ReadableMap settings, double maxNewTokens,
                       double seed, Promise promise) {
    run(promise, "GGML_GENERATE_FAILED", () -> {
      float[] sampling = {
        number(settings, "temperature", 1),
        number(settings, "topK", 0),
        number(settings, "topP", 1),
        number(settings, "minP", 0),
        number(settings, "repetitionPenalty", 1),
        number(settings, "repetitionWindow", Float.NaN),
        number(settings, "tokenRunPenalty", 0),
        number(settings, "tokenRunGrace", 10),
      };
      int[] ids = {
        (int) number(settings, "speechEnd", 0),
        (int) number(settings, "stop", 0),
        (int) number(settings, "audioStart", 0),
        (int) number(settings, "audioEnd", 0),
      };
      int[] tokens = nativeGenerate(ints(prompt), floats(speaker), sampling, ids, (int) maxNewTokens,
                                    (int) (long) seed);
      WritableArray out = Arguments.createArray();
      for (int token : tokens) out.pushInt(token);
      return out;
    });
  }

  @ReactMethod
  public void loadDecoder(String path, Promise promise) {
    run(promise, "GGML_LOAD_FAILED", () -> {
      nativeLoadDecoder(path);
      return null;
    });
  }

  /**
   * Codec tokens to 24 kHz float32 samples, returned as base64 bytes: a few
   * megabytes cross the bridge as one string much faster than as numbers.
   */
  @ReactMethod
  public void decode(ReadableArray tokens, ReadableArray promptToken, String promptFeat, ReadableArray embedding,
                     double seed, Promise promise) {
    run(promise, "GGML_DECODE_FAILED", () -> {
      byte[] featBytes = Base64.decode(promptFeat, Base64.DEFAULT);
      if (featBytes.length % (4 * 80) != 0) throw new IllegalArgumentException("promptFeat is not float32 mel frames.");
      float[] feat = new float[featBytes.length / 4];
      ByteBuffer.wrap(featBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(feat);
      byte[] samples = nativeDecode(ints(tokens), ints(promptToken), feat, floats(embedding), (int) (long) seed);
      return Base64.encodeToString(samples, Base64.NO_WRAP);
    });
  }

  /** Stops the generation in flight between tokens. */
  @ReactMethod
  public void cancel() {
    if (LOADED) nativeCancel();
  }

  @ReactMethod
  public void unload(Promise promise) {
    if (LOADED) nativeCancel();
    run(promise, "GGML_UNLOAD_FAILED", () -> {
      nativeUnload();
      return null;
    });
  }

  @Override
  public void invalidate() {
    if (LOADED) {
      nativeCancel();
      queue.execute(KittenTTS2GGMLModule::nativeUnload);
    }
    queue.shutdown();
    super.invalidate();
  }

  private interface Call {
    Object run() throws Exception;
  }

  private void run(Promise promise, String code, Call call) {
    if (!LOADED) {
      promise.reject("GGML_UNAVAILABLE", "This app was built without the SDK's ggml engine.");
      return;
    }
    queue.execute(() -> {
      try {
        promise.resolve(call.run());
      } catch (CancellationException error) {
        promise.reject("CANCELLED", "The request was cancelled.", error);
      } catch (Throwable error) {
        promise.reject(code, error.getMessage(), error);
      }
    });
  }

  private static float number(ReadableMap map, String key, float fallback) {
    return map.hasKey(key) && !map.isNull(key) ? (float) map.getDouble(key) : fallback;
  }

  private static int[] ints(ReadableArray array) {
    int[] out = new int[array.size()];
    for (int i = 0; i < out.length; i++) out[i] = array.getInt(i);
    return out;
  }

  private static float[] floats(ReadableArray array) {
    float[] out = new float[array.size()];
    for (int i = 0; i < out.length; i++) out[i] = (float) array.getDouble(i);
    return out;
  }

  private static native boolean nativeSupported();

  private static native String nativeDevice();

  private static native void nativeLoad(String path, int[] config, float[] floatConfig);

  private static native void nativeLoadDecoder(String path);

  private static native int[] nativeGenerate(int[] prompt, float[] speaker, float[] settings, int[] ids,
                                             int maxNewTokens, int seed);

  private static native byte[] nativeDecode(int[] tokens, int[] promptToken, float[] promptFeat, float[] embedding,
                                            int seed);

  private static native void nativeCancel();

  private static native void nativeUnload();
}
