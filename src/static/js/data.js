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
    if ($('fProject').value) p.set('project_id', $('fProject').value);
    if ($('fCorrected').checked) p.set('corrected_only', '1');
    if ($('fMissing').checked) p.set('missing_only', '1');
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
      return `
        <div class="tile ${state.selected.has(r.id) ? 'sel' : ''} ${r.active ? '' : 'inactive'}"
             data-idx="${i}" data-id="${r.id}">
          <input type="checkbox" class="cb" ${state.selected.has(r.id) ? 'checked' : ''}
                 aria-label="Select ${esc(r.filename)}">
          ${flag}
          ${img}
          <div class="tmeta">
            <div class="badges">
              <span class="tag ${esc(r.label)}">${esc(pretty(r.label))}</span>
              <span class="tag src">${esc(r.source)}</span>
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
    state.openIdx = idx;

    $('dFilename').textContent = r.filename;
    $('dFilename').title = r.image_path;
    const img = $('dImg');
    if (r.exists) {
      img.src = `/api/training-examples/${r.id}/image?full=1`;
      img.style.display = '';
    } else {
      img.removeAttribute('src');
      img.style.display = 'none';
    }

    const rows = [
      ['Class', pretty(r.label)],
      ['Imported as', r.original_label ? pretty(r.original_label) : '—'],
      ['Source', r.source],
      ['Project', r.project_name || '—'],
      ['In training', r.active ? 'yes' : 'no (excluded)'],
      ['Added', (r.created_at || '').slice(0, 19).replace('T', ' ')],
      ['Updated', (r.updated_at || '').slice(0, 19).replace('T', ' ')],
      ['Path', r.image_path],
    ];
    if (r.origin_ref) rows.push(['Origin', r.origin_ref]);
    $('dMeta').innerHTML = rows.map(([k, v]) =>
      `<div class="row"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`
    ).join('');

    $('dLabels').innerHTML = state.labels.map((cls, i) => `
      <button class="labelbtn ${cls === r.label ? 'on' : ''}" data-label="${esc(cls)}">
        ${esc(pretty(cls))}<kbd>${i + 1}</kbd>
      </button>`).join('');

    $('dNotes').value = r.notes || '';
    $('dToggle').textContent = r.active ? 'Exclude' : 'Restore';
    $('drawer').hidden = false;
  }

  function closeDrawer() {
    $('drawer').hidden = true;
    state.openIdx = -1;
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
      toast(`Set to ${pretty(label)}`);
      loadStats();
    } catch (e) {
      toast(e.message, true);
    }
  }

  /* ---------------- events ---------------- */

  $('grid').addEventListener('click', (ev) => {
    const tile = ev.target.closest('.tile');
    if (!tile) return;
    const id = Number(tile.dataset.id);
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
  ['fLabel', 'fSource', 'fProject', 'fActive', 'fCorrected', 'fMissing'].forEach((id) => {
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
  }

  /* keyboard: 1-9 relabel, j/k navigate, Esc close */
  document.addEventListener('keydown', (ev) => {
    if (ev.target.matches('input, textarea, select')) return;
    if (ev.key === 'Escape') { closeDrawer(); return; }
    if (state.openIdx < 0) return;
    if (ev.key === 'j') { openDrawer(Math.min(state.rows.length - 1, state.openIdx + 1)); return; }
    if (ev.key === 'k') { openDrawer(Math.max(0, state.openIdx - 1)); return; }
    const n = Number(ev.key);
    if (n >= 1 && n <= state.labels.length) {
      setLabel(state.rows[state.openIdx].id, state.labels[n - 1]);
    }
  });

  applyUrlFilters();
  refresh();
})();
