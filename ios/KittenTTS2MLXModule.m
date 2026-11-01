// React Native's view of KittenTTS2MLXEngine: Kitten TTS 2's language model on
// the GPU. Built only when the app turns MLX on (see KittenTTSReactNative.podspec).

#import <React/RCTBridgeModule.h>

#if __has_include(<KittenTTSReactNative/KittenTTSReactNative-Swift.h>)
#import <KittenTTSReactNative/KittenTTSReactNative-Swift.h>
#else
#import "KittenTTSReactNative-Swift.h"
#endif

@interface KittenTTS2MLXModule : NSObject <RCTBridgeModule>
@end

@implementation KittenTTS2MLXModule

RCT_EXPORT_MODULE(KittenTTS2MLX)

+ (BOOL)requiresMainQueueSetup
{
  return NO;
}

RCT_EXPORT_METHOD(isAvailable:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject)
{
  resolve(@(KittenTTS2MLXEngine.isAvailable));
}

RCT_EXPORT_METHOD(load:(NSString *)path
                  config:(NSDictionary *)config
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)
{
  [KittenTTS2MLXEngine.shared loadWithPath:path config:config completion:^(NSError *error) {
    if (error) {
      reject(@"MLX_LOAD_FAILED", error.localizedDescription, error);
    } else {
      resolve(nil);
    }
  }];
}

RCT_EXPORT_METHOD(generate:(NSArray<NSNumber *> *)prompt
                  speaker:(NSArray<NSNumber *> *)speaker
                  settings:(NSDictionary *)settings
                  maxNewTokens:(double)maxNewTokens
                  seed:(double)seed
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject)
{
  [KittenTTS2MLXEngine.shared generateWithPrompt:prompt
                                         speaker:speaker
                                        settings:settings
                                    maxNewTokens:(NSInteger)maxNewTokens
                                            seed:(uint32_t)seed
                                      completion:^(NSArray<NSNumber *> *tokens, NSError *error) {
    if (error) {
      reject(error.code == 2 ? @"CANCELLED" : @"MLX_GENERATE_FAILED", error.localizedDescription, error);
    } else {
      resolve(tokens);
    }
  }];
}

RCT_EXPORT_METHOD(cancel)
{
  [KittenTTS2MLXEngine.shared cancel];
}

RCT_EXPORT_METHOD(unload:(RCTPromiseResolveBlock)resolve reject:(RCTPromiseRejectBlock)reject)
{
  [KittenTTS2MLXEngine.shared unloadWithCompletion:^{
    resolve(nil);
  }];
}

@end
