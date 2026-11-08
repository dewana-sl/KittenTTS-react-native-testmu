// Runs the Kitten TTS 2 benchmark app on one TestMu (LambdaTest) real device and
// saves its report. TTS2_BASE_URL is where the device downloads the model files. One WebdriverIO session and no test runner: the runner's
// worker process died on start in GitHub Actions ("write EINVAL").
//
// The app shows its report as JSON while it runs, so a crash or a timeout still
// leaves the steps that finished, and why it stopped.
import fs from 'node:fs';
import path from 'node:path';
import { remote } from 'webdriverio';

const required = name => {
  if (!process.env[name]) throw new Error(`${name} must be set.`);
  return process.env[name];
};
const platformName = process.env.TESTMU_PLATFORM || 'iOS';
const deviceName = required('TESTMU_DEVICE');
const platformVersion = required('TESTMU_VERSION');
const app = required('TESTMU_APP_URL');
const base = required('TTS2_BASE_URL');
const REPORT = process.env.TTS2_REPORT || 'reports/tts2.json';
const TIMEOUT_MS = Number(process.env.TTS2_TIMEOUT_MS || 40 * 60 * 1000);
const BUNDLE_ID = 'dev.stellon.tts2bench';

function save(report) {
  fs.mkdirSync(path.dirname(REPORT), { recursive: true });
  fs.writeFileSync(REPORT, JSON.stringify({ deviceName, platformName, platformVersion, ...report }, null, 2));
}

async function reportText(driver) {
  const element = await driver.$('~tts2-report');
  if (!(await element.isExisting().catch(() => false))) return '';
  const names = platformName === 'iOS' ? ['label', 'value', 'name'] : ['text'];
  for (const read of [() => element.getText(), ...names.map(name => () => element.getAttribute(name))]) {
    const text = String((await read().catch(() => '')) || '');
    if (text.startsWith('{')) return text;
  }
  return '';
}

let driver;
try {
  driver = await remote({
    // As the TestMu CI's app sessions connect.
    hostname: 'mobile-hub.lambdatest.com',
    port: 80,
    path: '/wd/hub',
    user: required('LT_USERNAME'),
    key: required('LT_ACCESS_KEY'),
    logLevel: 'warn',
    // TestMu holds the request while the device is queued and the app installs,
    // which can take longer than the default 5 minutes. A request given up on is
    // still served later, and its session then holds the account's device.
    connectionRetryTimeout: 20 * 60 * 1000,
    connectionRetryCount: 0,
    capabilities: {
      platformName,
      'appium:deviceName': deviceName,
      'appium:platformVersion': platformVersion,
      'appium:app': app,
      'appium:autoAcceptAlerts': platformName === 'iOS',
      'appium:autoGrantPermissions': platformName === 'Android',
      'appium:newCommandTimeout': 900,
      'lt:options': {
        build: `Kitten TTS 2 on-device ${process.env.GITHUB_RUN_ID || 'local'}`,
        name: `Kitten TTS 2 ${platformName} - ${deviceName}`,
        project: 'KittenTTS React Native',
        isRealMobile: true,
        // Seconds without a command before TestMu ends the session; the runner polls every 15.
        idleTimeout: 180,
        console: true,
        devicelog: true,
        network: false,
        visual: false,
      },
    },
  });
} catch (error) {
  save({ done: false, error: `No TestMu session: ${error?.message ?? error}` });
  console.error(error);
  process.exit(1);
}

// What the device logged about the engines (which GPU, why not, GPU faults), and
// when the app stopped, about the app itself: crash and memory lines.
const ENGINE_LINES = /KittenTTS2|ggml|vulkan|Adreno|kgsl|Mali|DeviceLost|DEVICE_LOST|MLX|Fatal error|jetsam|lowmemorykiller|lmkd/i;
const APP_LINES = /tts2bench|EXC_|Terminat|FATAL|SIGABRT|SIGSEGV/i;
const CRASH_LINES = /ASI found|Fatal error|libc\+\+abi|terminating|Unable to build|program_source|DeviceLost|DEVICE_LOST|kgsl.*(fault|hang)|runs on|loaded on/i;
async function deviceLog(withApp) {
  try {
    const types = await driver.getLogTypes();
    const lines = [];
    for (const type of types.filter(t => ['crashlog', 'syslog', 'logcat'].includes(t))) {
      // Appium logs each report it reads, and the report names the engines: leave its lines out.
      const messages = (await driver.getLogs(type)).map(entry => String(entry.message)).filter(m => !/\bappium\b/i.test(m));
      const relevant = messages.filter(m => ENGINE_LINES.test(m) || (withApp && APP_LINES.test(m)));
      // The crash message itself first: later lines would push it out of the tail.
      const crash = messages.filter(m => CRASH_LINES.test(m)).slice(0, 30);
      lines.push(`[${type}] ${relevant.length} of ${messages.length} lines`, ...crash, '...', ...relevant.slice(-60));
    }
    return { types, lines };
  } catch (error) {
    return { error: String(error?.message ?? error) };
  }
}

const SWEEP = JSON.parse(process.env.TTS2_SWEEP || 'null') || [];
const FULL = process.env.TTS2_FULL !== 'false';
// The full run's mode, typed after the address: "auto" lets the SDK pick the engine.
const MODE = process.env.TTS2_MODE || '';
const SWEEP_TIMEOUT_MS = 6 * 60 * 1000;

/** Types the address into the app, taps Start and reads its report until it is done or stops. */
async function runApp(address, timeoutMs, onReport = () => {}) {
  const deadline = Date.now() + timeoutMs;
  let last = null;
  let stopped = '';
  try {
    // The app waits for the address of the model files.
    const input = await driver.$('~tts2-base');
    await input.waitForExist({ timeout: 120000 });
    await input.setValue(address);
    await driver.hideKeyboard().catch(() => {});
    await (await driver.$('~tts2-start')).click();
    while (Date.now() < deadline) {
      const text = await reportText(driver);
      if (text) {
        try {
          last = JSON.parse(text);
          onReport(last);
          console.log(`${new Date().toISOString()} ${last.status} (${(last.steps || []).length} steps)`);
        } catch {
          // A report caught mid-update; read it again next time.
        }
      }
      if (last?.done) break;
      const state = await driver.queryAppState(BUNDLE_ID).catch(() => -1);
      if (state === 1) {
        stopped = 'The app stopped running (crashed, or the system ended it for memory).';
        break;
      }
      await driver.pause(15000);
    }
  } catch (error) {
    stopped = `The session failed: ${error?.message ?? error}`;
  }
  if (!last?.done && !stopped) stopped = `No complete report within ${Math.round(timeoutMs / 60000)} minutes.`;
  return { last, stopped };
}

/** Starts the app again in a new process, with `env` ("NAME=value;...") in its environment (scripts/gpu-env.py). */
async function restartWith(env) {
  const extras = env ? [['s', 'ggml_env', env]] : [];
  try {
    await driver.execute('mobile: startActivity', { intent: `${BUNDLE_ID}/.MainActivity`, stop: true, wait: true, extras });
  } catch (error) {
    // Older UiAutomator2 drivers take these names instead.
    await driver.execute('mobile: startActivity', {
      appPackage: BUNDLE_ID, appActivity: '.MainActivity', dontStopAppOnReset: false,
      ...(env ? { optionalIntentArguments: `--es ggml_env "${env}"` } : {}),
    }).catch(() => { throw error; });
  }
}

let result = {};
let failed = false;
if (FULL) {
  const { last, stopped } = await runApp(MODE ? `${base}#${MODE}` : base, TIMEOUT_MS, report => save(report));
  if (last?.done) {
    result = { ...last, deviceLog: await deviceLog(false) };
  } else {
    failed = true;
    result = { ...(last || {}), done: false, error: `${stopped} Last status: ${last?.status ?? 'none'}`, deviceLog: await deviceLog(true) };
    console.error(stopped);
  }
  save(result);
}

// Short GPU checks, each in a new app process with different ggml-vulkan settings.
if (SWEEP.length) {
  result.sweep = [];
  for (const env of SWEEP) {
    console.log(`GPU check with ${env || 'no settings'}`);
    let entry;
    try {
      await restartWith(env);
      const { last, stopped } = await runApp(`${base}#gpu-check`, SWEEP_TIMEOUT_MS);
      entry = { env, gpu: last?.gpu, device: last?.device, steps: last?.steps ?? [], done: Boolean(last?.done) };
      if (!last?.done) Object.assign(entry, { error: `${stopped} Last status: ${last?.status ?? 'none'}`, deviceLog: await deviceLog(true) });
    } catch (error) {
      entry = { env, done: false, error: `Could not start the app: ${error?.message ?? error}` };
    }
    console.log(JSON.stringify(entry).slice(0, 2000));
    result.sweep.push(entry);
    save(result);
  }
  if (!FULL) {
    const first = result.sweep.find(entry => entry.done);
    result = { ...result, done: Boolean(first), gpu: first?.gpu, device: first?.device, backend: 'ggml' };
    save(result);
  }
}

await driver.deleteSession().catch(() => {});
process.exit(failed ? 1 : 0);
