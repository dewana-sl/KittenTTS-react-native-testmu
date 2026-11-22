// Markdown summary of the Kitten TTS 2 Nano / Pico / Mini benchmark reports in a folder.
//   node e2e/tiny/report.js reports/
const fs = require('fs');
const path = require('path');

const dir = process.argv[2] || 'reports';
const reports = fs.existsSync(dir)
  ? fs.readdirSync(dir).filter(f => f.endsWith('.json')).map(f => JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8')))
  : [];

const out = ['## Kitten TTS 2 Nano, Pico and Mini on real devices', ''];
out.push('RTF = seconds of work per second of audio (below 1 is faster than real time).', '');
out.push('| Asked for | Got | Model | create | RTF median (min-max) | Phonemize median | Stream: first chunk / RTF | Status |');
out.push('| --- | --- | --- | --- | --- | --- | --- | --- |');
for (const r of reports) {
  const asked = `${r.deviceName} (${r.platformName} ${r.platformVersion})`;
  const got = r.device ? `${r.device.brand ?? ''} ${r.device.model ?? ''}, ${r.device.os}, ${r.device.memoryGB ?? '?'} GB`.trim() : '-';
  if (!r.runs || r.runs.length === 0) {
    out.push(`| ${asked} | ${got} | - | - | - | - | - | ${r.error ? `❌ ${String(r.error).slice(0, 120)}` : r.status} |`);
    continue;
  }
  for (const run of r.runs) {
    const s = run.summary;
    const stream = run.stream ? `${(run.stream.firstChunkMs / 1000).toFixed(2)} s / ${run.stream.rtf}` : '-';
    const status = run.summary ? '✅' : `❌ ${String(r.error || r.status).slice(0, 120)}`;
    out.push(`| ${asked} | ${got} | ${run.model.replace('kitten-tts-2-', '')} ${run.weights} | ${run.createMs ?? '-'} ms | `
      + `${s ? `${s.rtfMedian} (${s.rtfMin}-${s.rtfMax})` : '-'} | ${s ? `${s.phonMsMedian} ms` : '-'} | ${stream} | ${status} |`);
  }
}
// The memory test: resident memory (MB) after each phase's first and last step, and the
// growth per step after the first (a leak grows every step; waiting garbage goes up and down).
out.push('', '### Memory (RSS, MB)', '');
out.push('| Got | Start | Nano create/dispose ×8: first → last (growth/cycle) | Mini + Japanese ×4 | Nano generate ×60 | Nano stream ×5 | End |');
out.push('| --- | --- | --- | --- | --- | --- | --- |');
for (const r of reports) {
  const m = r.memory?.summary;
  if (!m) continue;
  const got = r.device ? `${r.device.brand ?? ''} ${r.device.model ?? ''}`.trim() : r.deviceName;
  const cell = p => (m[p] ? `${m[p].firstMB} → ${m[p].lastMB} (${m[p].growthPerStepMB >= 0 ? '+' : ''}${m[p].growthPerStepMB})` : '-');
  out.push(`| ${got} | ${m.startMB} | ${cell('nano-cycle')} | ${cell('mini-ja-cycle')} | ${cell('nano-generate')} | ${cell('nano-stream')} | ${m.endMB} |`);
}
console.log(out.join('\n'));
