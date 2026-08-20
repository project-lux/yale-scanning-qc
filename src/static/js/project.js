/* Project detail: scan grid, plain-language review, run a pass, reprocess. */

(() => {
  const { get, post, toast, esc, $, pretty } = QC;

  const root = $('projectRoot');
  const PID = Number(root.dataset.projectId);
  const PAGE_TYPES = (root.dataset.pageTypes || '').split(',').filter(Boolean);

  const state = {
    page: 1,
    perPage: 60,
    rows: [],
    selected: new Set(),
    openIdx: -1,
    polling: null,
  };

  /* ---------------- loading ---------------- */

  function query() {
    const p = new URLSearchParams({ page: String(state.page), per_page: String(state.perPage) });
    if ($('fFlagged').value) p.set('flagged', $('fFlagged').value);
    if ($('fReviewed').value) p.set('reviewed', $('fReviewed').value);
    if ($('fPageType').value) p.set('page_type', $('fPageType').value);
    return p.toString();
  }

  async function load() {
    try {
      const d = await get(`/api/projects/${PID}/scans?${query()}`);
      state.rows = d.scans || [];
      renderCounts(d.counts || {});
      renderGrid();
      $('pageInfo').textContent = `page ${state.page} · ${state.rows.length} shown`;
      $('prevPage').disabled = state.page <= 1;
      $('nextPage').disabled = state.rows.length < state.perPage;
    } catch (e) {
      toast(e.message, true);
    }
  }

  function renderCounts(c) {
    $('counts').innerHTML = [
      `<span class="chip">scans <b>${c.total || 0}</b></span>`,
      `<span class="chip bad">flagged bad <b>${c.flagged || 0}</b></span>`,
      `<span class="chip good">clean <b>${c.clean || 0}</b></span>`,
      c.unknown ? `<span class="chip warn">unscored <b>${c.unknown}</b></span>` : '',
      `<span class="chip">reviewed <b>${c.reviewed || 0}</b></span>`,
      c.errors ? `<span class="chip warn">errors <b>${c.errors}</b></span>` : '',
      c.transcribed ? `<span class="chip">transcribed <b>${c.transcribed}</b></span>` : '',
    ].filter(Boolean).join('');
  }

  /* A review is stored as TP/FP/FN/TN; show it as what the human decided. */
  function reviewLabel(v) {
    if (v === 'true_positive' || v === 'false_negative') return 'bad';
    if (v === 'false_positive' || v === 'true_negative') return 'fine';
    return '';
  }

  function renderGrid() {
    const g = $('grid');
    if (!state.rows.length) {
      g.innerHTML = '';
      $('empty').hidden = false;
      return;
    }
    $('empty').hidden = true;
    g.innerHTML = state.rows.map((r, i) => {
      const img = r.has_jpeg || /\.(jpe?g|png)$/i.test(r.filepath)
        ? `<img class="thumb" loading="lazy" src="/api/projects/scans/${r.id}/image" alt="">`
        : '<div class="missing">no preview</div>';
      const verdict = r.bad_scan
        ? `<span class="tag ${esc(r.bad_scan)}">${esc(pretty(r.bad_scan))}` +
          `${r.bad_scan_conf ? ` ${(r.bad_scan_conf * 100).toFixed(0)}%` : ''}</span>`
        : '<span class="tag">unscored</span>';
      const rev = reviewLabel(r.review_verdict);
      return `
        <div class="tile ${state.selected.has(r.id) ? 'sel' : ''}" data-idx="${i}" data-id="${r.id}">
          <input type="checkbox" class="cb" ${state.selected.has(r.id) ? 'checked' : ''}
                 aria-label="Select ${esc(r.filename)}">
          ${rev ? `<span class="flag">${esc(rev)}</span>` : ''}
          ${img}
          <div class="tmeta">
            <div class="badges">
              ${verdict}
              ${r.page_type ? `<span class="tag src">${esc(pretty(r.page_type))}</span>` : ''}
            </div>
            <div class="fname" title="${esc(r.filepath)}">${esc(r.filename)}</div>
          </div>
        </div>`;
    }).join('');
    updateBulk();
  }

  function updateBulk() {
    const n = state.selected.size;
    $('bulkbar').hidden = n === 0;
    $('selCount').textContent = String(n);
    $('reprocessScope').textContent = n ? `${n} selected scan(s)` : 'whole project';
  }

  /* ---------------- drawer + review ---------------- */

  function openDrawer(idx) {
    const r = state.rows[idx];
    if (!r) return;
    state.openIdx = idx;

    $('dFilename').textContent = r.filename;
    $('dFilename').title = r.filepath;
    const img = $('dImg');
    if (r.has_jpeg || /\.(jpe?g|png)$/i.test(r.filepath)) {
      img.src = `/api/projects/scans/${r.id}/image?full=1`;
      img.style.display = '';
    } else {
      img.removeAttribute('src');
      img.style.display = 'none';
    }

    const rows = [
      ['Model says', r.bad_scan ? `${pretty(r.bad_scan)}` +
        (r.bad_scan_conf ? ` (${(r.bad_scan_conf * 100).toFixed(1)}%)` : '') : 'not scored'],
      ['Page type', r.page_type ? `${pretty(r.page_type)}${r.page_type_source ? ` · ${r.page_type_source}` : ''}` : '—'],
      ['Your review', reviewLabel(r.review_verdict) || 'not reviewed'],
      ['Pass', `${r.batch_label || '—'} · ${(r.batch_started || '').slice(0, 10)}`],
      ['Status', r.status + (r.error ? ` — ${r.error}` : '')],
      ['Path', r.filepath],
    ];
    if (r.transcription) rows.push(['Transcription', r.transcription.slice(0, 300)]);
    $('dMeta').innerHTML = rows.map(([k, v]) =>
      `<div class="row"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`
    ).join('');

    // Show which bookkeeping verdict the answer will record.
    const flagged = r.bad_scan === 'bad_scan';
    $('verdictHint').textContent = flagged
      ? 'The model flagged this one.'
      : 'The model did not flag this one.';
    const cur = reviewLabel(r.review_verdict);
    $('vBad').classList.toggle('on', cur === 'bad');
    $('vGood').classList.toggle('on', cur === 'fine');

    $('dPageType').value = r.review_page_type || '';
    $('dNotes').value = r.review_notes || '';
    $('drawer').hidden = false;
  }

  const closeDrawer = () => { $('drawer').hidden = true; state.openIdx = -1; };

  async function submitVerdict(isBad) {
    if (state.openIdx < 0) return;
    const r = state.rows[state.openIdx];
    try {
      const d = await post(`/api/projects/scans/${r.id}/verdict`, {
        is_bad: isBad,
        page_type: $('dPageType').value || null,
        notes: $('dNotes').value,
      });
      Object.assign(r, d.scan, {
        review_verdict: d.verdict,
        review_page_type: $('dPageType').value || null,
        review_notes: $('dNotes').value,
        filename: r.filename,
        has_jpeg: r.has_jpeg,
      });
      toast(`Recorded: ${isBad ? 'bad scan' : 'looks fine'} (${d.verdict.replace(/_/g, ' ')})`);
      renderGrid();
      // Move straight on — reviewing is a queue.
      if (state.openIdx < state.rows.length - 1) openDrawer(state.openIdx + 1);
      else openDrawer(state.openIdx);
    } catch (e) {
      toast(e.message, true);
    }
  }

  /* ---------------- run + reprocess ---------------- */

  function setRunning(on, label, detail, frac) {
    $('runstrip').hidden = !on;
    if (!on) return;
    $('runLabel').textContent = label || 'Working…';
    $('runDetail').textContent = detail || '';
    $('runBar').style.width = `${Math.round((frac || 0) * 100)}%`;
  }

  function startPolling() {
    if (state.polling) return;
    state.polling = setInterval(async () => {
      try {
        const d = await get(`/api/projects/${PID}/progress`);
        const p = d.progress || {};
        if (!d.running) {
          clearInterval(state.polling);
          state.polling = null;
          setRunning(false);
          toast('Finished');
          load();
          return;
        }
        const total = p.total || 0;
        const done = p.done || 0;
        setRunning(true,
          p.stage_rerun ? `Reprocessing (${p.stage_rerun})` : 'Running pass',
          `${done}/${total} · ${p.current || ''}`,
          total ? done / total : 0);
      } catch { /* transient; keep polling */ }
    }, 1200);
  }

  $('runBtn').addEventListener('click', async () => {
    const all = confirm('Run a pass on this project.\n\n'
      + 'OK = every file (re-process already-scanned files too)\n'
      + 'Cancel = new files only');
    try {
      await post(`/api/projects/${PID}/run`, { all });
      setRunning(true, 'Running pass', 'starting…', 0);
      startPolling();
    } catch (e) { toast(e.message, true); }
  });

  $('reprocessBtn').addEventListener('click', () => {
    $('reprocessPanel').hidden = !$('reprocessPanel').hidden;
  });
  $('cancelReprocess').addEventListener('click', () => { $('reprocessPanel').hidden = true; });
  $('bulkReprocess').addEventListener('click', () => { $('reprocessPanel').hidden = false; });

  $('doReprocess').addEventListener('click', async () => {
    const stages = [];
    if ($('stBadscan').checked) stages.push('badscan');
    if ($('stPagetype').checked) stages.push('pagetype');
    if ($('stTranscription').checked) stages.push('transcription');
    if (!stages.length) { toast('Pick at least one stage', true); return; }
    try {
      const body = { stages };
      if (state.selected.size) body.scan_ids = [...state.selected];
      const d = await post(`/api/projects/${PID}/rerun-stages`, body);
      $('reprocessPanel').hidden = true;
      toast(`Reprocessing ${d.scans} scan(s)`);
      setRunning(true, `Reprocessing (${stages.join('+')})`, 'starting…', 0);
      startPolling();
    } catch (e) { toast(e.message, true); }
  });

  $('cancelBtn').addEventListener('click', async () => {
    try { await post('/api/batches/cancel', {}); toast('Cancellation requested'); }
    catch (e) { toast(e.message, true); }
  });

  /* ---------------- events ---------------- */

  $('grid').addEventListener('click', (ev) => {
    const tile = ev.target.closest('.tile');
    if (!tile) return;
    const id = Number(tile.dataset.id);
    if (ev.target.classList.contains('cb')) {
      if (state.selected.has(id)) state.selected.delete(id);
      else state.selected.add(id);
      tile.classList.toggle('sel', state.selected.has(id));
      updateBulk();
      return;
    }
    openDrawer(Number(tile.dataset.idx));
  });

  $('vBad').addEventListener('click', () => submitVerdict(true));
  $('vGood').addEventListener('click', () => submitVerdict(false));
  $('dClose').addEventListener('click', closeDrawer);
  $('dPrev').addEventListener('click', () => openDrawer(Math.max(0, state.openIdx - 1)));
  $('dNext').addEventListener('click', () =>
    openDrawer(Math.min(state.rows.length - 1, state.openIdx + 1)));

  $('selAll').addEventListener('click', () => {
    state.rows.forEach((r) => state.selected.add(r.id));
    renderGrid();
  });
  $('selNone').addEventListener('click', () => { state.selected.clear(); renderGrid(); });

  ['fFlagged', 'fReviewed', 'fPageType'].forEach((id) =>
    $(id).addEventListener('change', () => { state.page = 1; load(); }));
  $('fPer').addEventListener('change', (ev) => {
    state.perPage = Number(ev.target.value) || 60;
    state.page = 1;
    load();
  });
  $('prevPage').addEventListener('click', () => {
    if (state.page > 1) { state.page--; closeDrawer(); load(); }
  });
  $('nextPage').addEventListener('click', () => { state.page++; closeDrawer(); load(); });

  document.addEventListener('keydown', (ev) => {
    if (ev.target.matches('input, textarea, select')) return;
    if (ev.key === 'Escape') { closeDrawer(); return; }
    if (state.openIdx < 0) return;
    if (ev.key === 'b') submitVerdict(true);
    else if (ev.key === 'g') submitVerdict(false);
    else if (ev.key === 'j') openDrawer(Math.min(state.rows.length - 1, state.openIdx + 1));
    else if (ev.key === 'k') openDrawer(Math.max(0, state.openIdx - 1));
  });

  /* page-type pickers */
  const opts = PAGE_TYPES.map((t) => `<option value="${esc(t)}">${esc(pretty(t))}</option>`).join('');
  $('fPageType').innerHTML = '<option value="">All</option>' + opts;
  $('dPageType').innerHTML = '<option value="">— unchanged —</option>' + opts;

  load();
  // A pass may already be running when the page opens.
  get(`/api/projects/${PID}/progress`).then((d) => { if (d.running) startPolling(); }).catch(() => {});
})();
