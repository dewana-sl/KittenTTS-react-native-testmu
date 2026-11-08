// Lets MLX run on iOS 17 (https://github.com/ml-explore/mlx-swift/issues/497).
//
// MLX builds its Metal kernels from source on the device. Since mlx 0.32.2 the
// matrix kernels use metal::remove_addrspace_t, which iOS 17's Metal library
// does not have, so the first matrix product fails to build. On iOS 17 only,
// this hands Metal the source with that name swapped for a stand-in that gives
// back the type it is given, which is what mlx used before 0.32.2. Newer iOS
// versions are left alone. Remove this once mlx-swift fixes the issue.

#import <Metal/Metal.h>
#import <objc/runtime.h>

#import "KittenTTS2MetalCompat.h"

// -[MTLDevice newLibraryWithSource:options:error:] returns an object the caller owns.
typedef id<MTLLibrary> __attribute__((ns_returns_retained)) (*NewLibraryIMP)(
    id, SEL, NSString *, MTLCompileOptions *, NSError *__autoreleasing *);

static NewLibraryIMP originalNewLibrary;

static NSString *const missingName = @"metal::remove_addrspace_t<";
static NSString *const standInName = @"kittentts2_remove_addrspace_t<";
static NSString *const standIn =
    @"template <typename T> struct kittentts2_remove_addrspace { typedef T type; };\n"
    @"template <typename T> using kittentts2_remove_addrspace_t = typename kittentts2_remove_addrspace<T>::type;\n";

NSString *KittenTTS2MetalCompatSource(NSString *source)
{
  if (![source containsString:missingName]) return source;
  return [standIn stringByAppendingString:[source stringByReplacingOccurrencesOfString:missingName
                                                                           withString:standInName]];
}

static id<MTLLibrary> __attribute__((ns_returns_retained)) newLibraryWithSource(
    id device, SEL selector, NSString *source, MTLCompileOptions *options, NSError *__autoreleasing *error)
{
  return originalNewLibrary(device, selector, KittenTTS2MetalCompatSource(source), options, error);
}

void KittenTTS2InstallMetalCompat(void)
{
  if (@available(iOS 18.0, *)) return;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return;
    Class deviceClass = [device class];
    SEL selector = @selector(newLibraryWithSource:options:error:);
    Method method = class_getInstanceMethod(deviceClass, selector);
    if (!method) return;
    originalNewLibrary = (NewLibraryIMP)method_getImplementation(method);
    // Override on the device's own class; replace in place if it defines the method itself.
    if (!class_addMethod(deviceClass, selector, (IMP)newLibraryWithSource, method_getTypeEncoding(method))) {
      method_setImplementation(method, (IMP)newLibraryWithSource);
    }
  });
}
