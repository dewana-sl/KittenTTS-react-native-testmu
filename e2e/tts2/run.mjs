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
    connectionRetryTimeout: 300000,
    connectionRetryCount: 2,
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
        idleTimeout: 1800,
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

// What the device logged about the app, for when it stops: crash and memory lines.
async function deviceLog() {
  try {
    const types = await driver.getLogTypes();
    const lines = [];
    for (const type of types.filter(t => ['crashlog', 'syslog', 'logcat'].includes(t))) {
      const messages = (await driver.getLogs(type)).map(entry => String(entry.message));
      const relevant = messages.filter(m => /tts2bench|jetsam|EXC_|Terminat|lowmemorykiller|lmkd|FATAL|SIGABRT|SIGSEGV|kittentts|MLX|ggml|Vulkan/i.test(m));
      lines.push(`[${type}] ${relevant.length} of ${messages.length} lines`, ...relevant.slice(-40));
    }
    return { types, lines };
  } catch (error) {
    return { error: String(error?.message ?? error) };
  }
}

const deadline = Date.now() + TIMEOUT_MS;
let last = null;
let stopped = '';
try {
  // The app waits for the address of the model files.
  const input = await driver.$('~tts2-base');
  await input.waitForExist({ timeout: 120000 });
  await input.setValue(base);
  await driver.hideKeyboard().catch(() => {});
  await (await driver.$('~tts2-start')).click();
  while (Date.now() < deadline) {
    const text = await reportText(driver);
    if (text) {
      try {
        last = JSON.parse(text);
        save(last);
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
if (!last?.done) {
  const reason = stopped || `No complete report within ${Math.round(TIMEOUT_MS / 60000)} minutes.`;
  const log = await deviceLog();
  save({ ...(last || {}), done: false, error: `${reason} Last status: ${last?.status ?? 'none'}`, deviceLog: log });
  console.error(reason);
  console.error(JSON.stringify(log, null, 2));
  await driver.deleteSession().catch(() => {});
  process.exit(1);
}
await driver.deleteSession().catch(() => {});
process.exit(0);
