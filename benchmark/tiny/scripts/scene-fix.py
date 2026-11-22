"""Move an Expo SDK 55 app to the UIScene life cycle, which apps built with the
iOS 27 SDK need or they are killed at launch: the window is created by a
SceneDelegate instead of the AppDelegate.

    python scene-fix.py <app folder>   (after expo prebuild --platform ios)
"""
import glob, plistlib, sys
root = sys.argv[1]
p = glob.glob(f'{root}/ios/*/AppDelegate.swift')[0]
s = open(p).read()
old = """#if os(iOS) || os(tvOS)
    window = UIWindow(frame: UIScreen.main.bounds)
    factory.startReactNative(
      withModuleName: "main",
      in: window,
      launchOptions: launchOptions)
#endif
"""
assert old in s
s = s.replace(old, "    // SceneDelegate creates the window: the iOS 27 SDK requires the scene life cycle.\n")
s += """
@objc(SceneDelegate)
class SceneDelegate: UIResponder, UIWindowSceneDelegate {
  var window: UIWindow?

  func scene(
    _ scene: UIScene,
    willConnectTo session: UISceneSession,
    options connectionOptions: UIScene.ConnectionOptions
  ) {
    guard let windowScene = scene as? UIWindowScene,
          let appDelegate = UIApplication.shared.delegate as? AppDelegate,
          let factory = appDelegate.reactNativeFactory else { return }
    let window = UIWindow(windowScene: windowScene)
    self.window = window
    appDelegate.window = window
    factory.startReactNative(withModuleName: "main", in: window, launchOptions: nil)
  }
}
"""
open(p, 'w').write(s)
ip = p.replace('AppDelegate.swift', 'Info.plist')
info = plistlib.load(open(ip, 'rb'))
info['UIApplicationSceneManifest'] = {
    'UIApplicationSupportsMultipleScenes': False,
    'UISceneConfigurations': {'UIWindowSceneSessionRoleApplication': [
        {'UISceneConfigurationName': 'Default Configuration', 'UISceneDelegateClassName': 'SceneDelegate'}]},
}
plistlib.dump(info, open(ip, 'wb'))
print('scene fix applied')
