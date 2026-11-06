// Joins every device's report into one Markdown summary.
//   node report.js <folder of JSON reports> > summary.md
const fs = require('fs');
const path = require('path');

const folder = process.argv[2];
const files = fs.existsSync(folder) ? fs.readdirSync(folder, { recursive: true }).filter(f => String(f).endsWith('.json')) : [];
const reports = files.map(f => JSON.parse(fs.readFileSync(path.join(folder, String(f)), 'utf8')))
  .sort((a, b) => String(a.platformName).localeCompare(String(b.platformName)) || String(a.deviceName).localeCompare(String(b.deviceName)));

const stepOf = (r, name) => (r.steps || []).find(s => s.name === name);
const fmt = (v, digits = 2) => (typeof v === 'number' ? v.toFixed(digits) : '–');
const cell = (r, name, pick) => {
  const s = stepOf(r, name);
  if (!s) return '–';
  return s.ok ? pick(s.detail) : '❌';
};

const lines = ['# Kitten TTS 2 on TestMu devices', ''];
lines.push('| Device | OS | Status | Engine | Download | Load | Whole RTF: default / 1 step / student | Decoder RTF, 19 s: default / 1 step / student | First streamed chunk | Cancel |');
lines.push('| --- | --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: |');
for (const r of reports) {
  const failed = (r.steps || []).filter(s => !s.ok);
  const status = r.done && failed.length === 0 && !r.error ? '✅' : '❌';
  const load = stepOf(r, 'load default');
  const whole = ['default', 'stock_1step', 'student_w4'].map(d => cell(r, `8 sentences ${d}`, x => `${fmt(x.rtf)}${x.audioOk ? '' : ' (bad audio)'}`)).join(' / ');
  const only = stepOf(r, 'decoder only');
  const decoder = only?.ok ? ['default', 'stock_1step', 'student_w4'].map(d => fmt(only.detail.median[d]?.longRtf, 3)).join(' / ') : (only ? '❌' : '–');
  const stream = cell(r, 'streaming, first chunk', x => `${fmt(x.chunks[0]?.seconds, 1)} s at ${fmt(x.chunks[0]?.atSeconds, 1)} s`);
  const cancel = cell(r, 'cancel', x => (x.cancelled ? `${x.afterMs} ms` : 'not cancelled'));
  lines.push(`| ${r.deviceName} | ${r.platformName} ${r.platformVersion} | ${status} | ${r.backend || '–'} ${r.device || ''} | ${load?.ok ? `${load.detail.downloadMB} MB, ${fmt(load.detail.downloadSeconds, 0)} s` : '–'} | ${load?.ok ? `${fmt(load.detail.loadSeconds, 1)} s` : '–'} | ${whole} | ${decoder} | ${stream} | ${cancel} |`);
}
const problems = reports.flatMap(r => [
  ...(r.error ? [`- **${r.deviceName}**: ${r.error}`] : []),
  ...(r.steps || []).filter(s => !s.ok).map(s => `- **${r.deviceName}**, ${s.name}: ${s.error}`),
]);
if (problems.length) lines.push('', '## What does not work', '', ...problems);
lines.push('', 'RTF is seconds of generating per second of speech. "Whole" is the 8 sentences end to end; "Decoder" is the audio decoder alone on the same 19 s of codec tokens (median of 4 rounds, decoders taking turns).');
process.stdout.write(`${lines.join('\n')}\n`);
