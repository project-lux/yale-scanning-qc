/* Records: the scan table, filterable on any column, exportable as CSV.
   Filtering, sorting and paging all happen on the server; the export link
   carries the same parameters, so the download is what is on screen. */

(() => {
  const { get, toast, esc, $ } = QC;

  const LS_COLS = 'qc-records-cols';
  const DEFAULT_SORT = { sort: 'processed_at', dir: 'desc' };

  const state = {
    cols: [],          // visible column keys, in schema order
    filters: {},       // column -> {in|contains|min|max}
    q: '',
    sort: DEFAULT_SORT.sort,
    dir: DEFAULT_SORT.dir,
    page: 1,
    perPage: 50,
    total: 0,
    rawOpen: false,
  };
  let schema = null;
  let colMap = {};
  let loadSeq = 0;

  /* ---------------- parameters ---------------- */

  const hasFilters = () => Object.keys(state.filters).length > 0;

  /* Everything that decides which rows and columns: shared by the table,
     the raw preview and the export link. */
  function baseParams() {
    const p = new URLSearchParams();
    if (hasFilters()) p.set('filters', JSON.stringify(state.filters));
    if (state.q) p.set('q', state.q);
    p.set('sort', state.sort);
    p.set('dir', state.dir);
    p.set('columns', state.cols.join(','));
    return p;
  }

  /* Filters, search and sort live in the address bar, so a view can be
     bookmarked, shared, or reached from a link on another page. */
  function writeUrl() {
    const p = new URLSearchParams();
    if (hasFilters()) p.set('filters', JSON.stringify(state.filters));
    if (state.q) p.set('q', state.q);
    if (state.sort !== DEFAULT_SORT.sort || state.dir !== DEFAULT_SORT.dir) {
      p.set('sort', state.sort);
      p.set('dir', state.dir);
    }
    const qs = p.toString();
    history.replaceState(null, '', qs ? `?${qs}` : location.pathname);
  }

  function readUrl() {
    const p = new URLSearchParams(location.search);
    try {
      const f = JSON.parse(p.get('filters') || '{}');
      // Drop anything this schema does not know instead of failing the page.
      state.filters = Object.fromEntries(
        Object.entries(f).filter(([k, v]) => colMap[k] && v && typeof v === 'object'));
    } catch { state.filters = {}; }
    state.q = p.get('q') || '';
    if (colMap[p.get('sort')]) state.sort = p.get('sort');
    if (p.get('dir') === 'asc' || p.get('dir') === 'desc') state.dir = p.get('dir');
  }

  /* ---------------- loading ---------------- */

  async function load() {
    const seq = ++loadSeq;
    writeUrl();
    renderChips();
    renderPresets();
    updateExport();
    const p = baseParams();
    p.set('page', String(state.page));
    p.set('per_page', String(state.perPage));
    try {
      const d = await get(`/api/records?${p}`);
      if (seq !== loadSeq) return;   // a newer request superseded this one
      state.total = d.total;
      renderTable(d.rows);
      renderSummary();
      updateExport();
      if (state.rawOpen) loadRaw();
    } catch (e) {
      if (seq === loadSeq) toast(e.message, true);
    }
  }

  async function loadRaw() {
    const p = baseParams();
    p.set('limit', '25');
    try {
      const res = await fetch(`/api/records/export.csv?${p}`);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const text = (await res.text()).replace(/^﻿/, '');
      $('rawCsv').textContent = text;
      const shown = Math.min(25, state.total);
      $('rawHint').textContent =
        `first ${shown} of ${state.total.toLocaleString()} row(s) · the download has all of them`;
    } catch (e) { toast(e.message, true); }
  }

  /* ---------------- table ---------------- */

  function cellHtml(col, v) {
    if (v === null || v === undefined || v === '') return '';
    const s = esc(v);
    if (col.key === 'result') {
      const tone = v === 'passed' ? 'no_bad_scan' : v === 'not_passed' ? 'bad_scan' : '';
      return `<span class="tag ${tone}">${s}</span>`;
    }
    if (col.key === 'model_outcome' && (v === 'false_alarm' || v === 'missed_bad')) {
      return `<span class="tag warn">${s}</span>`;
    }
    return s;
  }

  function renderTable(rows) {
    const cols = state.cols.map((k) => colMap[k]);
    const arrow = state.dir === 'asc' ? '▲' : '▼';
    const head = cols.map((c) => `
      <th data-key="${esc(c.key)}" class="${state.sort === c.key ? 'sorted' : ''}"
          title="${esc(c.desc)} Click to sort.">${esc(c.key)}${
            state.sort === c.key ? `<span class="arrow">${arrow}</span>` : ''}</th>`).join('');

    const body = rows.map((r) => `<tr>${cols.map((c) => {
      const v = r[c.key];
      const blank = v === null || v === undefined || v === '';
      // A value in a choice column is a one-click filter.
      const pick = c.kind === 'choice' && !blank ? ' pick' : '';
      const tip = c.kind === 'choice' && !blank
        ? `Filter to ${v}`
        : (typeof v === 'string' && v.length > 40 ? v.slice(0, 600) : '');
      return `<td class="k-${c.kind}${pick}${blank ? ' blank' : ''}" data-key="${esc(c.key)}"` +
        `${pick ? ` data-val="${esc(v)}"` : ''}${tip ? ` title="${esc(tip)}"` : ''}>` +
        `${cellHtml(c, v)}</td>`;
    }).join('')}</tr>`).join('');

    $('recTable').innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;
    $('rEmpty').hidden = rows.length > 0;
  }

  function renderSummary() {
    const from = state.total ? (state.page - 1) * state.perPage + 1 : 0;
    const to = Math.min(state.total, state.page * state.perPage);
    const narrowed = hasFilters() || state.q;
    $('rSummary').textContent = state.total
      ? `Rows ${from.toLocaleString()}–${to.toLocaleString()} of ${state.total.toLocaleString()}` +
        `${narrowed ? ` matching (of ${schema.total.toLocaleString()} in all)` : ''}`
      : 'No rows';
    $('rPage').textContent = `page ${state.page}`;
    $('rPrev').disabled = state.page <= 1;
    $('rNext').disabled = to >= state.total;
  }

  function updateExport() {
    const a = $('exportBtn');
    a.href = `/api/records/export.csv?${baseParams()}`;
    a.textContent = state.total || !schema
      ? `Export CSV (${(state.total || schema?.total || 0).toLocaleString()} rows)`
      : 'Export CSV';
    a.classList.toggle('disabled', !state.total && !!schema);
    a.setAttribute('aria-disabled', String(!state.total && !!schema));
  }

  /* ---------------- sort ---------------- */

  $('recTable').addEventListener('click', (ev) => {
    const th = ev.target.closest('th[data-key]');
    if (th) {
      const k = th.dataset.key;
      if (state.sort === k) state.dir = state.dir === 'asc' ? 'desc' : 'asc';
      else { state.sort = k; state.dir = 'asc'; }
      state.page = 1;
      load();
      return;
    }
    const td = ev.target.closest('td.pick');
    if (td) setChoice(td.dataset.key, [td.dataset.val]);
  });

  /* ---------------- filters ---------------- */

  function describe(key, spec) {
    const c = colMap[key];
    if (spec.in) {
      const vals = spec.in.map((v) => (v === '' ? '(blank)' : v));
      return `${c.key}: ${vals.join(', ')}`;
    }
    if (spec.contains) return `${c.key} contains “${spec.contains}”`;
    const lo = spec.min !== undefined ? `≥ ${spec.min}` : '';
    const hi = spec.max !== undefined ? `≤ ${spec.max}` : '';
    return `${c.key} ${[lo, hi].filter(Boolean).join(' and ')}`;
  }

  function renderChips() {
    const chips = Object.entries(state.filters).map(([k, spec]) =>
      `<span class="fchip" title="${esc(colMap[k].desc)}">${esc(describe(k, spec))}
         <button data-key="${esc(k)}" aria-label="Remove filter on ${esc(k)}">✕</button></span>`);
    if (state.q) {
      chips.push(`<span class="fchip">search “${esc(state.q)}”
        <button data-key="__q" aria-label="Clear search">✕</button></span>`);
    }
    if (chips.length > 1) chips.push('<button class="btn sm" id="clearAll">Clear all</button>');
    $('rChips').innerHTML = chips.join('');
  }

  $('rChips').addEventListener('click', (ev) => {
    if (ev.target.id === 'clearAll') {
      state.filters = {};
      state.q = '';
      $('rSearch').value = '';
    } else if (ev.target.dataset.key === '__q') {
      state.q = '';
      $('rSearch').value = '';
    } else if (ev.target.dataset.key) {
      delete state.filters[ev.target.dataset.key];
    } else return;
    state.page = 1;
    renderOp();
    load();
  });

  function setChoice(key, vals) {
    state.filters[key] = { in: vals };
    state.page = 1;
    renderOp();
    load();
  }

  /* The "add filter" control adapts to the column's kind. */
  function renderOp() {
    const col = colMap[$('rField').value];
    const cur = state.filters[col.key] || {};
    const box = $('rOp');
    if (col.kind === 'choice') {
      const f = schema.facets[col.key] || { values: [] };
      const sel = new Set(cur.in || []);
      box.innerHTML = `<div class="pickvals" id="rVals">${f.values.map((x) => `
        <label class="check"><input type="checkbox" value="${esc(x.value)}"
          ${sel.has(x.value) ? 'checked' : ''}>
          <span>${esc(x.value === '' ? '(blank)' : x.value)}
            <span class="hint">${x.count.toLocaleString()}</span></span></label>`).join('')}
        ${f.truncated ? '<p class="hint">Showing the most common values only.</p>' : ''}</div>`;
    } else if (col.kind === 'text') {
      box.innerHTML = `<label class="field"><span>contains</span>
        <input class="input" id="rText" value="${esc(cur.contains || '')}"
               placeholder="text to look for"></label>`;
    } else if (col.kind === 'number') {
      box.innerHTML = `
        <label class="field"><span>min</span>
          <input class="input num" id="rMin" type="number" step="any" value="${esc(cur.min ?? '')}"></label>
        <label class="field"><span>max</span>
          <input class="input num" id="rMax" type="number" step="any" value="${esc(cur.max ?? '')}"></label>`;
    } else {
      box.innerHTML = `
        <label class="field"><span>from</span>
          <input class="input" id="rMin" type="date" value="${esc(cur.min || '')}"></label>
        <label class="field"><span>to</span>
          <input class="input" id="rMax" type="date" value="${esc(cur.max || '')}"></label>`;
    }
  }

  function readOp(col) {
    if (col.kind === 'choice') {
      const vals = [...document.querySelectorAll('#rVals input:checked')].map((i) => i.value);
      return vals.length ? { in: vals } : null;
    }
    if (col.kind === 'text') {
      const t = $('rText').value.trim();
      return t ? { contains: t } : null;
    }
    const out = {};
    if ($('rMin').value !== '') out.min = col.kind === 'number' ? Number($('rMin').value) : $('rMin').value;
    if ($('rMax').value !== '') out.max = col.kind === 'number' ? Number($('rMax').value) : $('rMax').value;
    return Object.keys(out).length ? out : null;
  }

  $('rAdd').addEventListener('click', () => {
    const col = colMap[$('rField').value];
    const spec = readOp(col);
    if (!spec) {
      // Nothing entered: treat as removing this column's filter.
      if (state.filters[col.key]) { delete state.filters[col.key]; state.page = 1; load(); }
      else toast('Choose a value first', true);
      return;
    }
    state.filters[col.key] = spec;
    state.page = 1;
    load();
  });

  $('rField').addEventListener('change', renderOp);
  $('rOp').addEventListener('keydown', (ev) => { if (ev.key === 'Enter') $('rAdd').click(); });

  let searchTimer = null;
  $('rSearch').addEventListener('input', (ev) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      state.q = ev.target.value.trim();
      state.page = 1;
      load();
    }, 300);
  });

  /* ---------------- quick views ---------------- */

  function renderPresets() {
    const now = JSON.stringify(state.filters);
    $('presets').innerHTML = schema.presets.map((p) => `
      <button class="preset ${JSON.stringify(p.filters) === now && !state.q ? 'on' : ''}"
              data-id="${esc(p.id)}" title="${esc(p.hint)}">${esc(p.label)}</button>`).join('');
  }

  $('presets').addEventListener('click', (ev) => {
    const b = ev.target.closest('.preset');
    if (!b) return;
    const p = schema.presets.find((x) => x.id === b.dataset.id);
    state.filters = JSON.parse(JSON.stringify(p.filters));
    state.q = '';
    $('rSearch').value = '';
    state.page = 1;
    renderOp();
    load();
  });

  /* ---------------- columns ---------------- */

  function saveCols() {
    localStorage.setItem(LS_COLS, JSON.stringify(state.cols));
  }

  function renderColPanel() {
    const on = new Set(state.cols);
    const groups = [...new Set(schema.columns.map((c) => c.group))];
    $('colPanel').innerHTML = `
      <div class="colactions">
        <button class="btn sm" data-set="default">Default</button>
        <button class="btn sm" data-set="all">All</button>
      </div>` + groups.map((g) => `
      <div class="colgroup"><div class="lhead">${esc(g)}</div>${schema.columns
        .filter((c) => c.group === g).map((c) => `
        <label class="check" title="${esc(c.desc)}">
          <input type="checkbox" value="${esc(c.key)}" ${on.has(c.key) ? 'checked' : ''}>
          <span>${esc(c.key)}</span></label>`).join('')}</div>`).join('');
    $('colSummary').textContent = `Columns (${state.cols.length})`;
  }

  function setCols(keys) {
    const want = new Set(keys);
    const next = schema.columns.map((c) => c.key).filter((k) => want.has(k));
    if (!next.length) { toast('Keep at least one column', true); renderColPanel(); return; }
    state.cols = next;
    saveCols();
    renderColPanel();
    load();
  }

  $('colPanel').addEventListener('change', (ev) => {
    if (ev.target.type !== 'checkbox') return;
    setCols([...document.querySelectorAll('#colPanel input:checked')].map((i) => i.value));
  });
  $('colPanel').addEventListener('click', (ev) => {
    const b = ev.target.closest('button[data-set]');
    if (!b) return;
    setCols(b.dataset.set === 'all'
      ? schema.columns.map((c) => c.key)
      : schema.columns.filter((c) => c.default).map((c) => c.key));
  });
  // Close the picker when clicking elsewhere.
  document.addEventListener('click', (ev) => {
    const d = $('colPick');
    if (d.open && !d.contains(ev.target)) d.open = false;
  });

  /* ---------------- paging, raw CSV, export ---------------- */

  $('rPer').addEventListener('change', (ev) => {
    state.perPage = Number(ev.target.value) || 50;
    state.page = 1;
    load();
  });
  $('rPrev').addEventListener('click', () => { if (state.page > 1) { state.page--; load(); } });
  $('rNext').addEventListener('click', () => { state.page++; load(); });

  $('rawBtn').addEventListener('click', () => {
    state.rawOpen = !state.rawOpen;
    $('rawPanel').hidden = !state.rawOpen;
    $('rawBtn').textContent = state.rawOpen ? 'Hide raw CSV' : 'Show raw CSV';
    $('rawBtn').setAttribute('aria-expanded', String(state.rawOpen));
    if (state.rawOpen) loadRaw();
  });

  $('exportBtn').addEventListener('click', (ev) => {
    if (!state.total) { ev.preventDefault(); toast('There are no rows to export', true); }
  });

  /* ---------------- start ---------------- */

  async function init() {
    try {
      schema = await get('/api/records/schema');
    } catch (e) { toast(e.message, true); return; }
    colMap = Object.fromEntries(schema.columns.map((c) => [c.key, c]));

    let saved = [];
    try { saved = JSON.parse(localStorage.getItem(LS_COLS) || '[]'); } catch { /* ignore */ }
    saved = saved.filter((k) => colMap[k]);
    state.cols = saved.length ? saved : schema.columns.filter((c) => c.default).map((c) => c.key);

    $('rField').innerHTML = [...new Set(schema.columns.map((c) => c.group))].map((g) =>
      `<optgroup label="${esc(g)}">${schema.columns.filter((c) => c.group === g)
        .map((c) => `<option value="${esc(c.key)}">${esc(c.key)}</option>`).join('')}</optgroup>`).join('');

    // Start on the column people reach for first: did it pass?
    if (colMap.result) $('rField').value = 'result';

    readUrl();
    $('rSearch').value = state.q;
    renderOp();
    renderColPanel();
    load();
  }
  init();
})();
