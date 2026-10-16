/* Settings editor. Fields are grouped so related knobs sit together, and
   anything the server accepts but we have not grouped still shows up. */

(() => {
  const { get, post, toast, esc, $ } = QC;

  const GROUPS = [
    ['Training data', [
      ['neg_sample_ratio', 'Clean pages per bad scan',
       'Every bad scan and every past model mistake is always included. Clean '
       + 'pages are sampled to this multiple of the bad count, spread evenly '
       + 'across projects.'],
      ['neg_sample_seed', 'Sampling seed',
       'Leave blank to draw a different random sample of clean pages on every '
       + 'build. Set a number to make builds reproducible.'],
      ['preserve_agreements', 'Also keep scans the model got right',
       'The watched folder is a staging area whose files move on. By default '
       + 'only scans the annotator says the model got wrong are copied into '
       + 'the app and kept as training data (0). Set 1 to keep every reviewed '
       + 'scan.'],
    ]],
    ['Training', [
      ['retrain_epochs', 'Default epochs', ''],
      ['retrain_auto_activate', 'Activate new model automatically', '1 = yes, 0 = no'],
    ]],
    ['Schedule', [
      ['schedule_enabled', 'Daily pass enabled', '1 = yes, 0 = no'],
      ['schedule_time', 'Run at (HH:MM, local)', ''],
    ]],
    ['Page types', [
      ['custom_page_types', 'Extra classes', 'Comma-separated.'],
      ['transcribe_page_types', 'Transcribe these page types', 'Comma-separated.'],
    ]],
    ['Transcription', [
      ['transcription_backend', 'Backend',
       'off | sglang | vertex | kraken (local OCR with line polygons) | '
       + 'nara (local Qwen VLM)'],
      ['kraken_model_path', 'Kraken model file',
       'Path to a kraken recognition model (.safetensors), e.g. the '
       + 'PP-OCRv6 medium model.'],
      ['kraken_device', 'Kraken device', 'mps | cpu | auto'],
      ['nara_model_id', 'NARA VLM model',
       'HuggingFace id or local directory of the page-transcription VLM.'],
      ['nara_server_url', 'NARA vLLM server',
       'Blank = run locally. "spark:8061" = vLLM on the DGX via the SSH '
       + 'tunnel. Or a full http(s) URL.'],
      ['transcribe_concurrency', 'Pages at a time',
       'Concurrent transcription requests against a server backend '
       + '(vLLM/SGLang/Vertex). Local engines always run one at a time.'],
      ['sglang_base_url', 'SGLang URL', ''],
      ['sglang_model', 'SGLang model', ''],
      ['sglang_api_key', 'SGLang API key', ''],
      ['vertex_project', 'Vertex project', ''],
      ['vertex_location', 'Vertex location', "must be 'global'"],
      ['vertex_model', 'Vertex model', ''],
    ]],
  ];

  let current = {};

  function field(key, label, help) {
    const v = current[key] ?? '';
    return `<label class="field block">
      <span>${esc(label)} <code class="hint">${esc(key)}</code></span>
      <input class="input" style="width:100%" data-key="${esc(key)}" value="${esc(v)}">
      ${help ? `<span class="hint">${esc(help)}</span>` : ''}
    </label>`;
  }

  async function load() {
    try {
      const d = await get('/api/settings');
      current = d.settings || {};
      const grouped = new Set(GROUPS.flatMap(([, fs]) => fs.map(([k]) => k)));
      let html = GROUPS.map(([title, fields]) => `
        <section class="panel">
          <h2 class="sectionh" style="margin-top:0">${esc(title)}</h2>
          ${fields.map(([k, l, h]) => field(k, l, h)).join('')}
        </section>`).join('');

      const rest = Object.keys(current).filter((k) => !grouped.has(k)).sort();
      if (rest.length) {
        html += `<section class="panel">
          <h2 class="sectionh" style="margin-top:0">Other</h2>
          ${rest.map((k) => field(k, k, '')).join('')}</section>`;
      }
      $('groups').innerHTML = html;
    } catch (e) { toast(e.message, true); }
  }

  $('saveBtn').addEventListener('click', async () => {
    const updates = {};
    document.querySelectorAll('input[data-key]').forEach((el) => {
      if (el.value !== (current[el.dataset.key] ?? '')) updates[el.dataset.key] = el.value;
    });
    if (!Object.keys(updates).length) { toast('Nothing changed'); return; }
    try {
      const d = await post('/api/settings', updates);
      current = d.settings || {};
      toast(`Saved ${Object.keys(updates).length} setting(s)`);
    } catch (e) { toast(e.message, true); }
  });

  load();
})();
