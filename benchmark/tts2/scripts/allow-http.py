"""Let the Android release build fetch over plain HTTP: the model files come from
the runner through the LambdaTest tunnel at http://localhost.lambdatest.com.
Test only. Also builds arm64 only, which every TestMu Android device is.

    python allow-http.py <app folder>   (after expo prebuild --platform android)
"""
import pathlib, re, sys

android = pathlib.Path(sys.argv[1]) / "android"
manifest = android / "app/src/main/AndroidManifest.xml"
text = manifest.read_text()
if "android:usesCleartextTraffic=" in text:
    text = re.sub(r'android:usesCleartextTraffic="[^"]*"', 'android:usesCleartextTraffic="true"', text)
else:
    text = text.replace("<application ", '<application android:usesCleartextTraffic="true" ', 1)
manifest.write_text(text)
gradle = android / "app/build.gradle"
gradle.write_text(re.sub(r'usesCleartextTraffic\s*:\s*"false"', 'usesCleartextTraffic: "true"', gradle.read_text()))
properties = android / "gradle.properties"
props = re.sub(r"(?m)^reactNativeArchitectures=.*$", "reactNativeArchitectures=arm64-v8a", properties.read_text())
properties.write_text(props)
print([line for line in props.splitlines() if line.startswith(("reactNativeArchitectures", "kittentts."))])
