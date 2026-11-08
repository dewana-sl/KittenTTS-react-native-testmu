"""Let the runner start the Android app with engine settings ("NAME=value,..."),
in the process's environment before React Native starts, so the engine and
ggml-vulkan read them when they set up the GPU. The app saves them to
files/ggml_env_next.txt when Appium types "x#set-env#<settings>"; the next start
reads and deletes that file. A launch intent's "ggml_env" extra works too, but
TestMu's Appium did not deliver one. The process writes its id and the settings
it got to files/ggml_env.txt, which the app puts in its report: a setting only
counts in a new process. "$FILES" in a value is the app's files folder. Test only, for trying settings on TestMu's phones.
"""
import pathlib, re, sys

java = pathlib.Path(sys.argv[1]) / "android/app/src/main/java"
activity = next(java.rglob("MainActivity.kt"))
text = activity.read_text()
if "ggml_env" not in text:
    hook = (
        "\n    val nextEnv = java.io.File(filesDir, \"ggml_env_next.txt\")\n"
        "    val ggmlEnv = (intent?.getStringExtra(\"ggml_env\") ?: if (nextEnv.exists()) nextEnv.readText() else \"\").trim()\n"
        "    nextEnv.delete()\n"
        "    ggmlEnv.split(',', ';').filter { it.contains('=') }.forEach {\n"
        "      val (name, value) = it.split('=', limit = 2)\n"
        "      android.system.Os.setenv(name.trim(), value.trim().replace(\"\\$FILES\", filesDir.path), true)\n"
        "    }\n"
        "    java.io.File(filesDir, \"ggml_env.txt\").writeText(\"${android.os.Process.myPid()} $ggmlEnv\")"
    )
    text, count = re.subn(r"(override fun onCreate\(savedInstanceState: Bundle\?\) \{)", lambda m: m.group(1) + hook, text, count=1)
    if count != 1:
        sys.exit(f"No onCreate in {activity}")
    activity.write_text(text)
print(f"{activity}: ggml_env hook in place")
