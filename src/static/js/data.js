/* Training-data browser: inspect, correct, exclude and delete labeled images. */

(() => {
  const { get, post, toast, esc, $, pretty } = QC;

  const state = {
    role: 'badscan',
    page: 1,
    perPage: 60,
    labels: [],
    rows: [],
    selected: new Set(),
    openIdx: -1,
    similar: null,      // {id, filename} while ranking by likeness to one example
    minScore: 0,
  };

  /* ---------------- data loading ---------------- */

  function filterQuery() {
    const p = new URLSearchParams({
      role: state.role,
      page: String(state.page),
      per_page: String(state.perPage),
      active: $('fActive').value,
    });
    if ($('fLabel').value) p.set('label', $('fLabel').value);
    if ($('fSource').value) p.set('source', $('fSource').value);
    if ($('fVerdict').value) p.set('verdict', $('fVerdict').value);
    if ($('fProject').value) p.set('project_id', $('fProject').value);
    if ($('fCorrected').checked) p.set('corrected_only', '1');
    if ($('fMissing').checked) p.set('missing_only', '1');
    if (state.similar) {
      p.set('similar_to', String(state.similar.id));
      if (state.minScore) p.set('min_score', String(state.minScore));
    }
    return p.toString();
  }

  async function loadStats() {
    const s = await get(`/api/training-examples/stats?role=${state.role}`);
    state.labels = s.labels || [];

    const chips = [];
    for (const cls of state.labels) {
      const n = s.by_label[cls] || 0;
      const tone = cls === 'bad_scan' ? 'bad' : cls === 'no_bad_scan' ? 'good' : '';
      chips.push(`<span class="chip ${tone}">${esc(pretty(cls))} <b>${n}</b></span>`);
    }
    // Classes present in the data but not in the role's class list.
    for (const [cls, n] of Object.entries(s.by_label)) {
      if (!state.labels.includes(cls)) {
        chips.push(`<span class="chip warn">${esc(pretty(cls))} <b>${n}</b> (not a class)</span>`);
      }
    }
    chips.push(`<span class="chip">in training <b>${s.total_active}</b></span>`);
    for (const [src, n] of Object.entries(s.by_source)) {
      chips.push(`<span class="chip src">${esc(src)} <b>${n}</b></span>`);
    }
    if (s.model_errors) {
      chips.push(`<span class="chip kept" title="Scans an annotator said the model got wrong — always in the dataset">model errors <b>${s.model_errors}</b></span>`);
    }
    if (s.corrected) chips.push(`<span class="chip warn">corrected <b>${s.corrected}</b></span>`);
    if (s.inactive) chips.push(`<span class="chip warn">excluded <b>${s.inactive}</b></span>`);
    $('statbar').innerHTML = chips.join('');

    // Keep the class/source pickers in sync with what actually exists.
    fillSelect($('fLabel'), Object.keys(s.by_label), 'All');
    fillSelect($('fSource'), Object.keys(s.by_source), 'All');
    fillSelect($('bulkLabel'), state.labels, '—');
  }

  function fillSelect(sel, values, allLabel) {
    const keep = sel.value;
    sel.innerHTML = `<option value="">${esc(allLabel)}</option>` +
      values.map((v) => `<option value="${esc(v)}">${esc(pretty(v))}</option>`).join('');
    if (values.includes(keep)) sel.value = keep;
  }

  async function loadGrid() {
    const data = await get(`/api/training-examples?${filterQuery()}`);
    state.rows = data.examples || [];
    if (data.labels) state.labels = data.labels;
    renderSimbar(data.similarity);
    renderGrid();
    $('pageInfo').textContent = `page ${state.page} · ${state.rows.length} shown`;
    $('prevPage').disabled = state.page <= 1;
    $('nextPage').disabled = state.rows.length < state.perPage;
  }

  async function refresh() {
    try {
      await Promise.all([loadStats(), loadGrid()]);
    } catch (e) {
      toast(e.message, true);
    }
  }

  /* ---------------- similar-image mode ---------------- */

  function renderSimbar(st) {
    $('simbar').hidden = !state.similar;
    if (!state.similar) return;
    $('simName').textContent = state.similar.filename;
    $('simThumb').src = `/api/training-examples/${state.similar.id}/image`;
    let msg = `${state.rows.length} shown, best match first`;
    if (st && st.running) {
      msg += ` · indexing ${st.done}/${st.todo} new images — results fill in as it runs`;
    }
    $('simStatus').textContent = msg;
  }

  function findSimilar(r) {
    if (!r) return;
    state.similar = { id: r.id, filename: r.filename };
    state.page = 1;
    // Stay in review mode, opened on the query image (always ranked first).
    state.openIdx = -1;
    loadGrid().then(() => {
      const i = state.rows.findIndex((x) => x.id === r.id);
      if (i >= 0) openDrawer(i); else closeDrawer();
    }).catch((e) => toast(e.message, true));
  }

  function exitSimilar() {
    state.similar = null;
    state.page = 1;
    closeDrawer();
    loadGrid().catch((e) => toast(e.message, true));
  }

  $('simExit').addEventListener('click', exitSimilar);
  $('simSelect').addEventListener('click', () => {
    state.rows.forEach((r) => state.selected.add(r.id));
    renderGrid();
  });
  $('simMin').addEventListener('input', (ev) => {
    const v = Number(ev.target.value);
    $('simMinVal').textContent = v ? `${v}%` : 'any';
  });
  $('simMin').addEventListener('change', (ev) => {
    state.minScore = Number(ev.target.value) / 100;
    state.page = 1;
    loadGrid().catch((e) => toast(e.message, true));
  });

  /* ---------------- grid ---------------- */

  function renderGrid() {
    const g = $('grid');
    if (!state.rows.length) {
      g.innerHTML = '';
      $('empty').hidden = false;
      return;
    }
    $('empty').hidden = true;
    g.innerHTML = state.rows.map((r, i) => {
      const img = r.exists
        ? `<img class="thumb" loading="lazy" src="/api/training-examples/${r.id}/image" alt="">`
        : `<div class="missing">file missing<br><small>${esc(r.filename)}</small></div>`;
      const flag = r.corrected ? '<span class="flag">edited</span>' : '';
      const isQuery = state.similar && state.similar.id === r.id;
      const sim = r.similarity != null
        ? `<span class="tag sim" title="Visual similarity">${(r.similarity * 100).toFixed(0)}%</span>` : '';
      const err = (r.verdict === 'false_positive' || r.verdict === 'false_negative')
        ? `<span class="tag kept" title="The model said ${pretty(r.meta && r.meta.model_bad_scan || '?')}; the annotator disagreed">${esc(pretty(r.verdict))}</span>` : '';
      return `
        <div class="tile ${state.selected.has(r.id) ? 'sel' : ''} ${r.active ? '' : 'inactive'} ${i === state.openIdx ? 'open' : ''} ${isQuery ? 'query' : ''}"
             data-idx="${i}" data-id="${r.id}">
          <input type="checkbox" class="cb" ${state.selected.has(r.id) ? 'checked' : ''}
                 aria-label="Select ${esc(r.filename)}">
          ${flag}
          <button class="simbtn" title="Show images that look like this one">≈ Similar</button>
          ${img}
          <div class="tmeta">
            <div class="badges">
              ${sim}
              <span class="tag ${esc(r.label)}">${esc(pretty(r.label))}</span>
              <span class="tag src">${esc(r.source)}</span>
              ${err}
            </div>
            <div class="fname" title="${esc(r.image_path)}">${esc(r.filename)}</div>
          </div>
        </div>`;
    }).join('');
    updateBulkBar();
  }

  function updateBulkBar() {
    const n = state.selected.size;
    $('bulkbar').hidden = n === 0;
    $('selCount').textContent = String(n);
  }

  /* ---------------- drawer ---------------- */

  function openDrawer(idx) {
    const r = state.rows[idx];
    if (!r) return;
    const wasOpen = state.openIdx >= 0;
    state.openIdx = idx;
    markOpenTile();

    $('dFilename').textContent = r.filename;
    $('dFilename').title = r.image_path;
    const img = $('dImg');
    $('dImgWrap').classList.remove('actual');
    if (r.exists) {
      img.src = `/api/training-examples/${r.id}/image?full=1`;
      img.style.display = '';
    } else {
      img.removeAttribute('src');
      img.style.display = 'none';
    }

    const m = r.meta || {};
    const pct = (v) => (v != null && v !== '' ? ` (${(Number(v) * 100).toFixed(1)}%)` : '');
    const rows = [
      ['Class', pretty(r.label)],
      ['Imported as', r.original_label ? pretty(r.original_label) : '—'],
      ['Source', r.source],
      ['Project', r.project_name || m.project_name || '—'],
      ['In training', r.active ? 'yes' : 'no (excluded)'],
    ];
    if (r.verdict || m.model_bad_scan) {
      // What the model said when the annotator answered — the reason this
      // image is here at all.
      rows.push(['Model said', m.model_bad_scan
        ? `${pretty(m.model_bad_scan)}${pct(m.model_bad_scan_conf)}` +
          (m.badscan_model ? ` · ${m.badscan_model}` : '')
        : '—']);
      if (m.model_page_type) {
        rows.push(['Model page type', `${pretty(m.model_page_type)}` +
          (m.model_page_type_source ? ` · ${m.model_page_type_source}` : '')]);
      }
      rows.push(['Outcome', r.verdict ? pretty(r.verdict) : (m.review_page_type ? 'page type corrected' : '—')]);
      rows.push(['Reviewed by', `${m.reviewer || '—'}` +
        (m.reviewed_at ? ` · ${String(m.reviewed_at).slice(0, 16).replace('T', ' ')}` : '')]);
      if (m.review_notes) rows.push(['Review notes', m.review_notes]);
      if (m.batch) rows.push(['Pass', m.batch]);
    }
    rows.push(['Added', (r.created_at || '').slice(0, 19).replace('T', ' ')]);
    rows.push(['Updated', (r.updated_at || '').slice(0, 19).replace('T', ' ')]);
    if (m.original_path || r.origin_ref) rows.push(['Original file', m.original_path || r.origin_ref]);
    rows.push(['Kept copy', r.image_path]);
    $('dMeta').innerHTML = rows.map(([k, v]) =>
      `<div class="row"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`
    ).join('');

    $('dLabels').innerHTML = state.labels.map((cls, i) => `
      <button class="labelbtn ${cls === r.label ? 'on' : ''}" data-label="${esc(cls)}">
        ${esc(pretty(cls))}<kbd>${i + 1}</kbd>
      </button>`).join('');

    $('dNotes').value = r.notes || '';
    $('dToggle').textContent = r.active ? 'Exclude' : 'Restore';
    loadHistory(r);
    $('drawer').hidden = false;
    $('workspace').classList.add('reviewing');
    if (!wasOpen) $('workspace').scrollIntoView({ block: 'start' });
    const tile = $('grid').querySelector('.tile.open');
    if (tile) tile.scrollIntoView({ block: 'nearest' });
  }

  /* Everything an annotator has done to this image, newest first. */
  async function loadHistory(r) {
    const box = $('dHistory');
    box.innerHTML = '';
    let rows;
    try {
      rows = (await get(`/api/training-examples/${r.id}/history`)).history || [];
    } catch { return; }
    if (state.rows[state.openIdx] !== r) return;
    box.innerHTML = '<div class="lhead">History</div>' +
      (rows.length ? rows.map(historyRow).join('') : '<p class="hint">No annotations logged yet.</p>');
  }

  function reviewLabel(v) {
    if (v === 'true_positive' || v === 'false_negative') return 'bad';
    if (v === 'false_positive' || v === 'true_negative') return 'fine';
    return '';
  }

  function historyRow(h) {
    const when = (h.created_at || '').slice(0, 16).replace('T', ' ');
    let what;
    if (h.action === 'verdict') {
      what = `answered <b>${esc(reviewLabel(h.new_value) || h.new_value)}</b>` +
        ` <span class="hint">(${esc(pretty(h.new_value))})</span>`;
    } else if (h.action === 'page_type') {
      what = `page type <b>${esc(pretty(h.new_value) || '—')}</b>` +
        (h.old_value ? ` <span class="hint">was ${esc(pretty(h.old_value))}</span>` : '');
    } else if (h.action === 'relabel') {
      what = `training label <b>${esc(pretty(h.new_value))}</b>` +
        ` <span class="hint">was ${esc(pretty(h.old_value))}</span>`;
    } else if (h.action === 'notes') {
      what = 'edited notes';
    } else {
      what = esc(h.action);
    }
    return `<div class="hrow">
      <span class="hwhen">${esc(when)}</span>
      <span class="hwho">${esc(h.reviewer || '—')}</span>
      <span class="hwhat">${what}${h.notes ? `<div class="hnote">“${esc(h.notes)}”</div>` : ''}</span>
    </div>`;
  }

  function closeDrawer() {
    const wasIdx = state.openIdx;
    $('drawer').hidden = true;
    $('workspace').classList.remove('reviewing');
    state.openIdx = -1;
    markOpenTile();
    // Land back on the tile that was being reviewed.
    const tile = $('grid').querySelector(`.tile[data-idx="${wasIdx}"]`);
    if (tile) tile.scrollIntoView({ block: 'center' });
  }

  function markOpenTile() {
    $('grid').querySelectorAll('.tile.open').forEach((t) => t.classList.remove('open'));
    const t = $('grid').querySelector(`.tile[data-idx="${state.openIdx}"]`);
    if (t) t.classList.add('open');
  }

  async function setLabel(id, label) {
    try {
      const { example } = await post(`/api/training-examples/${id}`, { label });
      // Merge the server's row back in so "edited" flags stay accurate.
      const i = state.rows.findIndex((r) => r.id === id);
      if (i >= 0) {
        state.rows[i] = {
          ...state.rows[i], ...example,
          filename: state.rows[i].filename,
          exists: state.rows[i].exists,
          corrected: !!(example.original_label && example.label !== example.original_label),
        };
        renderGrid();
        if (state.openIdx === i) openDrawer(i);
      }
      toast(`Set to ${pretty(label)}${example.scan_id ? ' · review updated to match' : ''}`);
      loadStats();
      if (state.openIdx === i) loadHistory(state.rows[i]);
    } catch (e) {
      toast(e.message, true);
    }
  }

  /* ---------------- events ---------------- */

  $('grid').addEventListener('click', (ev) => {
    const tile = ev.target.closest('.tile');
    if (!tile) return;
    const id = Number(tile.dataset.id);
    if (ev.target.classList.contains('simbtn')) {
      findSimilar(state.rows[Number(tile.dataset.idx)]);
      return;
    }
    if (ev.target.classList.contains('cb')) {
      if (state.selected.has(id)) state.selected.delete(id);
      else state.selected.add(id);
      tile.classList.toggle('sel', state.selected.has(id));
      updateBulkBar();
      return;
    }
    openDrawer(Number(tile.dataset.idx));
  });

  $('dLabels').addEventListener('click', (ev) => {
    const btn = ev.target.closest('.labelbtn');
    if (!btn || state.openIdx < 0) return;
    setLabel(state.rows[state.openIdx].id, btn.dataset.label);
  });

  $('dClose').addEventListener('click', closeDrawer);
  $('dSimilar').addEventListener('click', () => findSimilar(state.rows[state.openIdx]));
  $('dImgWrap').addEventListener('click', () => $('dImgWrap').classList.toggle('actual'));
  $('dPrev').addEventListener('click', () => openDrawer(Math.max(0, state.openIdx - 1)));
  $('dNext').addEventListener('click', () =>
    openDrawer(Math.min(state.rows.length - 1, state.openIdx + 1)));

  $('dSave').addEventListener('click', async () => {
    if (state.openIdx < 0) return;
    const r = state.rows[state.openIdx];
    try {
      await post(`/api/training-examples/${r.id}`, { notes: $('dNotes').value });
      r.notes = $('dNotes').value;
      toast('Notes saved');
      loadHistory(r);
    } catch (e) { toast(e.message, true); }
  });

  $('dToggle').addEventListener('click', async () => {
    if (state.openIdx < 0) return;
    const r = state.rows[state.openIdx];
    try {
      await post(`/api/training-examples/${r.id}`, { active: !r.active });
      toast(r.active ? 'Excluded from training' : 'Restored to training');
      closeDrawer();
      refresh();
    } catch (e) { toast(e.message, true); }
  });

  $('dDelete').addEventListener('click', async () => {
    if (state.openIdx < 0) return;
    const r = state.rows[state.openIdx];
    const owned = r.source === 'synthetic' || r.source === 'curated';
    const msg = owned
      ? `Delete this example permanently?\n\nAlso delete the image file (${r.source} copy)?\n\nOK = delete row and file, Cancel = keep the file.`
      : 'Delete this example permanently? The original scan file is never touched.';
    if (owned) {
      if (!confirm('Delete this training example?')) return;
      const alsoFile = confirm(msg);
      await doDelete([r.id], alsoFile);
    } else {
      if (!confirm(msg)) return;
      await doDelete([r.id], false);
    }
    closeDrawer();
  });

  async function doDelete(ids, removeFiles) {
    try {
      const r = await post('/api/training-examples/delete', { ids, remove_files: removeFiles });
      toast(`Deleted ${r.deleted}${r.files_removed ? `, removed ${r.files_removed} file(s)` : ''}`);
      state.selected.clear();
      refresh();
    } catch (e) { toast(e.message, true); }
  }

  /* bulk */
  $('selAll').addEventListener('click', () => {
    state.rows.forEach((r) => state.selected.add(r.id));
    renderGrid();
  });
  $('selNone').addEventListener('click', () => {
    state.selected.clear();
    renderGrid();
  });
  $('bulkLabel').addEventListener('change', async (ev) => {
    const label = ev.target.value;
    if (!label || !state.selected.size) return;
    if (!confirm(`Set ${state.selected.size} example(s) to "${pretty(label)}"?`)) {
      ev.target.value = '';
      return;
    }
    try {
      const r = await post('/api/training-examples/bulk',
        { ids: [...state.selected], label });
      toast(`Relabeled ${r.updated}`);
      state.selected.clear();
      refresh();
    } catch (e) { toast(e.message, true); }
    ev.target.value = '';
  });
  $('bulkExclude').addEventListener('click', () => bulkActive(false));
  $('bulkRestore').addEventListener('click', () => bulkActive(true));

  async function bulkActive(active) {
    if (!state.selected.size) return;
    try {
      const r = await post('/api/training-examples/bulk',
        { ids: [...state.selected], active });
      toast(`${active ? 'Restored' : 'Excluded'} ${r.updated}`);
      state.selected.clear();
      refresh();
    } catch (e) { toast(e.message, true); }
  }

  $('bulkDelete').addEventListener('click', async () => {
    const n = state.selected.size;
    if (!n) return;
    if (!confirm(`Permanently delete ${n} training example(s)?`)) return;
    const alsoFiles = confirm(
      'Also delete the underlying image files?\n\n' +
      'Only synthetic and curated copies are removed — original scans are never touched.\n\n' +
      'OK = delete files too, Cancel = keep files.');
    await doDelete([...state.selected], alsoFiles);
  });

  /* filters */
  ['fLabel', 'fSource', 'fVerdict', 'fProject', 'fActive', 'fCorrected', 'fMissing'].forEach((id) => {
    $(id).addEventListener('change', () => { state.page = 1; refresh(); });
  });
  $('fPer').addEventListener('change', (ev) => {
    state.perPage = Number(ev.target.value) || 60;
    state.page = 1;
    loadGrid().catch((e) => toast(e.message, true));
  });
  $('roleSel').addEventListener('change', (ev) => {
    state.role = ev.target.value;
    state.page = 1;
    state.similar = null;  // example ids are per role
    state.selected.clear();
    closeDrawer();
    refresh();
  });

  $('prevPage').addEventListener('click', () => {
    if (state.page > 1) { state.page--; closeDrawer(); loadGrid(); }
  });
  $('nextPage').addEventListener('click', () => {
    state.page++; closeDrawer(); loadGrid();
  });

  $('syncBtn').addEventListener('click', async () => {
    const btn = $('syncBtn');
    btn.disabled = true;
    btn.textContent = 'Syncing…';
    try {
      const r = await post('/api/training-examples/sync', { role: state.role });
      const a = r.added || {};
      toast(a.total ? `Added ${a.total} new example(s)` : 'Already up to date');
      refresh();
    } catch (e) { toast(e.message, true); }
    btn.disabled = false;
    btn.textContent = 'Sync sources';
  });

  /* Deep link from a project card: /data?project=<id>&role=<role> */
  function applyUrlFilters() {
    const q = new URLSearchParams(location.search);
    const project = q.get('project');
    if (project && [...$('fProject').options].some((o) => o.value === project)) {
      $('fProject').value = project;
    }
    const role = q.get('role');
    if (role && [...$('roleSel').options].some((o) => o.value === role)) {
      state.role = role;
      $('roleSel').value = role;
    }
    const verdict = q.get('verdict');
    if (verdict && [...$('fVerdict').options].some((o) => o.value === verdict)) {
      $('fVerdict').value = verdict;
    }
  }

  /* keyboard: 1-9 relabel, ←/→ (or k/j) navigate, Esc close */
  document.addEventListener('keydown', (ev) => {
    if (ev.target.matches('input, textarea, select')) return;
    if (ev.key === 'Escape') {
      if (state.openIdx >= 0) closeDrawer();
      else if (state.similar) exitSimilar();
      return;
    }
    if (state.openIdx < 0) return;
    if (ev.key === 's') { findSimilar(state.rows[state.openIdx]); return; }
    if (ev.key === 'ArrowRight' || ev.key === 'j') {
      ev.preventDefault();
      openDrawer(Math.min(state.rows.length - 1, state.openIdx + 1));
      return;
    }
    if (ev.key === 'ArrowLeft' || ev.key === 'k') {
      ev.preventDefault();
      openDrawer(Math.max(0, state.openIdx - 1));
      return;
    }
    const n = Number(ev.key);
    if (n >= 1 && n <= state.labels.length) {
      setLabel(state.rows[state.openIdx].id, state.labels[n - 1]);
    }
  });

  applyUrlFilters();
  refresh();
})();
