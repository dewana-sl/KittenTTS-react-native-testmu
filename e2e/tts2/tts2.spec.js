// Waits for the benchmark app to finish and saves its report. The app shows the
// report as JSON while it runs, so a crash or a timeout still leaves the steps
// that finished, and the reason it stopped.
const fs = require('fs');
const path = require('path');

const REPORT = process.env.TTS2_REPORT || 'reports/tts2.json';
const TIMEOUT_MS = Number(process.env.TTS2_TIMEOUT_MS || 40 * 60 * 1000);
const BUNDLE_ID = 'dev.stellon.tts2bench';

async function reportText() {
  const element = await $('~tts2-report');
  if (!(await element.isExisting().catch(() => false))) return '';
  const names = driver.isIOS ? ['label', 'value', 'name'] : ['text'];
  for (const read of [() => element.getText(), ...names.map(name => () => element.getAttribute(name))]) {
    const text = String((await read().catch(() => '')) || '');
    if (text.startsWith('{')) return text;
  }
  return '';
}

function save(report) {
  fs.mkdirSync(path.dirname(REPORT), { recursive: true });
  fs.writeFileSync(REPORT, JSON.stringify({
    deviceName: process.env.TESTMU_DEVICE,
    platformName: process.env.TESTMU_PLATFORM,
    platformVersion: process.env.TESTMU_VERSION,
    ...report,
  }, null, 2));
}

describe('Kitten TTS 2 on the device', () => {
  it('runs the benchmark app to the end', async () => {
    const deadline = Date.now() + TIMEOUT_MS;
    let last = null;
    let stopped = '';
    while (Date.now() < deadline) {
      const text = await reportText();
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
      await browser.pause(15000);
    }
    if (!last?.done) {
      const reason = stopped || `No complete report within ${Math.round(TIMEOUT_MS / 60000)} minutes.`;
      save({ ...(last || {}), done: false, error: `${reason} Last status: ${last?.status ?? 'none'}` });
      throw new Error(reason);
    }
  });
});
