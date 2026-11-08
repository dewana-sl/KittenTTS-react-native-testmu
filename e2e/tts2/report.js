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
// Metal's compile errors run to pages of warnings: keep the first line and the first error.
const short = text => {
  const lines = String(text).split('\n');
  const error = lines.slice(1).find(line => /error:/.test(line));
  const kept = [lines[0], ...(error ? [error.trim()] : [])].join(' … ');
  return kept.length > 400 ? `${kept.slice(0, 400)}…` : kept;
};
const problems = reports.flatMap(r => [
  ...(r.error ? [`- **${r.deviceName}**: ${short(r.error)}`] : []),
  ...(r.steps || []).filter(s => !s.ok).map(s => `- **${r.deviceName}**, ${s.name}: ${short(s.error)}`),
]);
if (problems.length) lines.push('', '## What does not work', '', ...problems);

const gpus = reports.filter(r => r.gpu);
if (gpus.length) lines.push('', '## Where ggml runs', '', ...gpus.map(r => `- **${r.deviceName}** (${r.model || '?'}): ${r.gpu}`));

// The short GPU checks: one app start per set of ggml-vulkan settings.
const swept = reports.filter(r => r.sweep?.length);
if (swept.length) {
  lines.push('', '## GPU settings tried', '', '| Device | Settings | Process (id, settings it got) | Runs on | RTF | Decoder RTF | Audio (rms, zero crossings) | Error |', '| --- | --- | --- | --- | ---: | ---: | --- | --- |');
  for (const r of swept) {
    for (const e of r.sweep) {
      const one = (e.steps || []).find(s => s.name === 'one sentence');
      const failedStep = (e.steps || []).find(s => !s.ok);
      const audio = one?.ok ? `${one.detail.audioOk ? 'ok' : 'bad'} (${one.detail.rms[0]}, ${one.detail.zcr?.[0] ?? '–'})` : '–';
      const error = e.error || failedStep?.error || '';
      lines.push(`| ${r.deviceName} | ${e.env || 'none'} | ${e.process?.native || '–'} | ${e.device || '–'} | ${one?.ok ? fmt(one.detail.rtf) : '–'} | ${one?.ok ? fmt(one.detail.decoderRtf, 3) : '–'} | ${audio} | ${short(error).replace(/\|/g, '/').replace(/\n/g, ' ')} |`);
    }
  }
}
lines.push('', 'RTF is seconds of generating per second of speech. "Whole" is the 8 sentences end to end; "Decoder" is the audio decoder alone on the same 19 s of codec tokens (median of 4 rounds, decoders taking turns).');
process.stdout.write(`${lines.join('\n')}\n`);
