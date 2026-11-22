// Runs the Kitten TTS 2 Nano / Pico / Mini benchmark app on one TestMu (LambdaTest) real
// device and saves its report. The app starts on its own and shows its report as JSON; this
// script polls it. One WebdriverIO session and no test runner: the runner's worker process
// died on start in GitHub Actions ("write EINVAL").
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
const REPORT = process.env.TINY_REPORT || 'reports/tiny.json';
const TIMEOUT_MS = Number(process.env.TINY_TIMEOUT_MS || 40 * 60 * 1000);
const BUNDLE_ID = 'dev.stellon.tinybench';

function save(report) {
  fs.mkdirSync(path.dirname(REPORT), { recursive: true });
  fs.writeFileSync(REPORT, JSON.stringify({ deviceName, platformName, platformVersion, ...report }, null, 2));
}

async function reportText(driver) {
  const element = await driver.$('~tiny-report');
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
    hostname: 'mobile-hub.lambdatest.com',
    port: 80,
    path: '/wd/hub',
    user: required('LT_USERNAME'),
    key: required('LT_ACCESS_KEY'),
    logLevel: 'warn',
    // TestMu holds the request while the device is queued and the app installs.
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
        build: `Kitten TTS 2 Nano Pico Mini ${process.env.GITHUB_RUN_ID || 'local'}`,
        name: `Kitten TTS 2 small models ${platformName} - ${deviceName}`,
        project: 'KittenTTS React Native',
        isRealMobile: true,
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

const APP_LINES = /tinybench|KittenTTS|ReactNativeJS|EXC_|Terminat|FATAL|SIGABRT|SIGSEGV|jetsam|lowmemorykiller|lmkd/i;
async function deviceLog() {
  try {
    const types = await driver.getLogTypes();
    const lines = [];
    for (const type of types.filter(t => ['crashlog', 'syslog', 'logcat'].includes(t))) {
      const messages = (await driver.getLogs(type)).map(entry => String(entry.message)).filter(m => !/\bappium\b/i.test(m));
      const relevant = messages.filter(m => APP_LINES.test(m));
      lines.push(`[${type}] ${relevant.length} of ${messages.length} lines`, ...relevant.slice(-80));
    }
    return { types, lines };
  } catch (error) {
    return { error: String(error?.message ?? error) };
  }
}

let last = null;
let stopped = '';
const deadline = Date.now() + TIMEOUT_MS;
try {
  while (Date.now() < deadline) {
    const text = await reportText(driver);
    if (text) {
      try {
        last = JSON.parse(text);
        save(last);
        console.log(`${new Date().toISOString()} ${last.status}`);
      } catch {
        // A report caught mid-update; read it again next time.
      }
    }
    if (last?.done) break;
    const state = await driver.queryAppState(BUNDLE_ID).catch(() => -1);
    if (state === 1 && last) {
      stopped = 'The app stopped running (crashed, or the system ended it for memory).';
      break;
    }
    await driver.pause(15000);
  }
} catch (error) {
  stopped = `The session failed: ${error?.message ?? error}`;
}
if (!last?.done && !stopped) stopped = `No complete report within ${Math.round(TIMEOUT_MS / 60000)} minutes.`;
const failed = !last?.done || Boolean(last?.error);
save({ ...(last || {}), ...(last?.done ? {} : { done: false, error: `${stopped} Last status: ${last?.status ?? 'none'}` }),
  ...(failed ? { deviceLog: await deviceLog() } : {}) });
await driver.deleteSession().catch(() => {});
process.exit(failed ? 1 : 0);
