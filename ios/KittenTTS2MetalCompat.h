#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// On iOS 17, fixes MLX's Metal kernel source as it is built (see KittenTTS2MetalCompat.m).
/// Does nothing on later versions. Safe to call more than once.
FOUNDATION_EXPORT void KittenTTS2InstallMetalCompat(void);

/// The source with iOS 17's missing Metal name swapped for a stand-in. Exposed for tests.
FOUNDATION_EXPORT NSString *KittenTTS2MetalCompatSource(NSString *source);

NS_ASSUME_NONNULL_END
