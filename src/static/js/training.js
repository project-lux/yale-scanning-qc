/* Live training: start a run, stream progress over SSE, chart it.
   Charts are hand-drawn SVG so the page needs no external library. */

(() => {
  const { get, post, toast, esc, $ } = QC;

  const state = { role: 'badscan', es: null, history: [] };

  /* ---------------- tiny SVG line chart ---------------- */

  function lineChart(el, series, opts = {}) {
    const W = 520, H = 170, P = { t: 10, r: 12, b: 24, l: 44 };
    const pts = series.filter((s) => s.values.some((v) => v !== null && v !== undefined));
    if (!pts.length) { el.innerHTML = ''; return; }

    const xs = pts[0].values.length;
    const all = pts.flatMap((s) => s.values.filter((v) => typeof v === 'number'));
    let lo = Math.min(...all), hi = Math.max(...all);
    if (opts.zeroBased) lo = Math.min(0, lo);
    if (hi === lo) { hi = lo + 1; }
    const pad = (hi - lo) * 0.08;
    lo -= pad; hi += pad;

    const x = (i) => P.l + (xs <= 1 ? 0 : (i / (xs - 1)) * (W - P.l - P.r));
    const y = (v) => P.t + (1 - (v - lo) / (hi - lo)) * (H - P.t - P.b);

    // horizontal gridlines + labels
    const ticks = 4;
    let grid = '';
    for (let k = 0; k <= ticks; k++) {
      const v = lo + ((hi - lo) * k) / ticks;
      const yy = y(v);
      grid += `<line x1="${P.l}" y1="${yy}" x2="${W - P.r}" y2="${yy}" class="grid"/>`
        + `<text x="${P.l - 6}" y="${yy + 3}" class="axis" text-anchor="end">`
        + `${opts.pct ? (v * 100).toFixed(0) + '%' : v.toFixed(v < 1 ? 3 : 2)}</text>`;
    }
    // x labels: first and last epoch
    grid += `<text x="${P.l}" y="${H - 6}" class="axis">1</text>`
      + `<text x="${W - P.r}" y="${H - 6}" class="axis" text-anchor="end">${xs}</text>`;

    const paths = pts.map((s) => {
      let d = '', open = false;
      s.values.forEach((v, i) => {
        if (typeof v !== 'number') { open = false; return; }
        d += `${open ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)} `;
        open = true;
      });
      const last = [...s.values].reverse().find((v) => typeof v === 'number');
      const dot = typeof last === 'number'
        ? `<circle cx="${x(s.values.length - 1)}" cy="${y(last)}" r="3" fill="${s.color}"/>`
        : '';
      return `<path d="${d.trim()}" fill="none" stroke="${s.color}" stroke-width="2"/>${dot}`;
    }).join('');

    const legend = pts.map((s, i) =>
      `<g transform="translate(${P.l + i * 110},${P.t + 2})">
         <rect width="9" height="9" rx="2" fill="${s.color}"/>
         <text x="13" y="8" class="axis">${esc(s.name)}</text></g>`).join('');

    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="linechart" role="img"
      aria-label="${esc(opts.label || 'chart')}">${grid}${paths}${legend}</svg>`;
  }

  function drawCharts() {
    const h = state.history;
    const has = h.length > 0;
    $('noMetrics').hidden = has;
    if (!has) { $('lossChart').innerHTML = ''; $('accChart').innerHTML = ''; return; }

    const pick = (k) => h.map((e) => (typeof e[k] === 'number' ? e[k] : null));
    const hasAny = (k) => h.some((e) => typeof e[k] === 'number');

    const lossSeries = [];
    if (hasAny('train_loss')) lossSeries.push({ name: 'train', values: pick('train_loss'), color: '#00356b' });
    if (hasAny('val_loss')) lossSeries.push({ name: 'val', values: pick('val_loss'), color: '#b42318' });
    lineChart($('lossChart'), lossSeries, { label: 'training loss', zeroBased: true });

    const accSeries = [];
    if (hasAny('top1')) accSeries.push({ name: 'top-1', values: pick('top1'), color: '#16794a' });
    // top-5 is pinned at 100% for a binary classifier, so it says nothing.
    const top5 = pick('top5');
    if (hasAny('top5') && top5.some((v) => typeof v === 'number' && v < 0.999)) {
      accSeries.push({ name: 'top-5', values: top5, color: '#92600a' });
    }
    lineChart($('accChart'), accSeries, { label: 'accuracy', pct: true });

    const last = h[h.length - 1] || {};
    $('lossLatest').textContent = typeof last.train_loss === 'number'
      ? `latest ${last.train_loss.toFixed(4)}` : '';
    $('accLatest').textContent = typeof last.top1 === 'number'
      ? `latest ${(last.top1 * 100).toFixed(1)}%` : '';
  }

  /* ---------------- live stream ---------------- */

  function renderRun(p, running) {
    const active = running || (p && p.version);
    // Keep a finished run visible so its curves can still be read.
    if (!active && !state.history.length) { $('runpanel').hidden = true; return; }
    if (!active) return;
    $('runpanel').hidden = false;
    if (!p || !p.version) return;
    $('runTitle').textContent = `${p.version || p.role} ← ${p.dataset || ''}`;
    const counts = p.counts ? Object.entries(p.counts).map(([k, v]) => `${k} ${v}`).join(' · ') : '';
    $('runSub').textContent = [p.status, counts].filter(Boolean).join(' — ');
    const eps = p.epochs || 0;
    // Ultralytics fires a final fit callback past the requested count.
    const ep = eps ? Math.min(p.epoch || 0, eps) : (p.epoch || 0);
    $('runEpoch').textContent = eps ? `epoch ${ep}/${eps}` : `epoch ${ep}`;
    $('runWhere').textContent = p.where === 'remote' ? 'GPU box' : (p.where || '—');
    $('runBar').style.width = eps ? `${Math.round((ep / eps) * 100)}%` : '0%';
    $('runSpin').style.visibility = running ? 'visible' : 'hidden';

    state.history = Array.isArray(p.history) ? p.history : [];
    drawCharts();
  }

  function openStream() {
    if (state.es) state.es.close();
    const es = new EventSource('/api/training/stream');
    state.es = es;

    es.onmessage = (ev) => {
      try {
        const d = JSON.parse(ev.data);
        renderRun(d.progress || {}, d.running);
        $('startBtn').disabled = !!d.running;
      } catch { /* ignore a malformed frame */ }
    };
    es.addEventListener('done', () => {
      es.close();
      state.es = null;
      $('startBtn').disabled = false;
      $('runSpin').style.visibility = 'hidden';
      $('runSub').textContent = 'finished';
      toast('Training finished');
      // Deliberately do not reload state here: that would clear the panel
      // and throw away the curves the user just watched.
      loadRuns();
      loadModels();
    });
    es.addEventListener('timeout', () => { es.close(); state.es = null; });
    es.onerror = () => {
      // EventSource retries on its own; only stop if the run is over.
      if (!state.es) return;
    };
  }

  /* ---------------- loads ---------------- */

  async function loadState() {
    try {
      const d = await get('/api/training/state');
      const r = d.remote || {};
      $('gpubar').innerHTML = [
        r.reachable
          ? `<span class="chip good">GPU box up${r.gpu ? ` · ${esc(r.gpu)}` : ''}</span>`
          : '<span class="chip warn">GPU box unreachable — training will run locally</span>',
        r.cuda ? '<span class="chip">CUDA</span>' : '',
        ...Object.entries(d.active_models || {}).map(([role, m]) =>
          `<span class="chip">${esc(role)}: <b>${esc(m ? m.version : 'none')}</b></span>`),
        ...Object.entries(d.datasets || {}).map(([role, s]) =>
          `<span class="chip">${esc(role)} store <b>${s.total}</b></span>`),
      ].filter(Boolean).join('');
      $('startBtn').disabled = !!d.running;
      if (d.running) { renderRun(d.progress || {}, true); openStream(); }
    } catch (e) { toast(e.message, true); }
  }

  async function loadRuns() {
    try {
      const { runs } = await get('/api/training/runs?limit=20');
      $('runsTable').querySelector('tbody').innerHTML = (runs || []).map((r) => {
        let acc = '—';
        try {
          const m = JSON.parse(r.model_metrics || '{}');
          if (typeof m.top1_accuracy === 'number') acc = `${(m.top1_accuracy * 100).toFixed(1)}%`;
        } catch { /* seed rows carry no metrics */ }
        const ds = r.dataset_version
          || (() => { try { return JSON.parse(r.dataset || '{}').version || '—'; } catch { return '—'; } })();
        return `<tr data-run="${r.id}" class="clickable" title="Show this run's curves">
          <td>#${r.id}${r.model_version ? ` <span class="hint">${esc(r.model_version)}</span>` : ''}</td>
          <td>${esc(r.role)}</td><td>${esc(ds)}</td>
          <td>${r.epochs ?? '—'}</td><td>${acc}</td>
          <td><span class="tag ${r.status === 'done' ? 'no_bad_scan' : r.status === 'error' ? 'bad_scan' : ''}">${esc(r.status)}</span>
            ${r.error ? `<div class="hint">${esc(String(r.error).slice(0, 120))}</div>` : ''}</td>
          <td>${esc((r.started_at || '').slice(0, 19).replace('T', ' '))}</td>
        </tr>`;
      }).join('') || '<tr><td colspan="7" class="empty">No runs yet.</td></tr>';
    } catch (e) { toast(e.message, true); }
  }

  async function loadModels() {
    try {
      const { models } = await get('/api/training/models');
      $('modelsTable').querySelector('tbody').innerHTML = (models || []).map((m) => {
        let acc = '—', ds = '—';
        try {
          const mm = JSON.parse(m.metrics || '{}');
          if (typeof mm.top1_accuracy === 'number') acc = `${(mm.top1_accuracy * 100).toFixed(1)}%`;
          ds = mm.dataset_version || '—';
        } catch { /* seed rows have no metrics */ }
        return `<tr>
          <td><b>${esc(m.version)}</b>${m.is_active ? ' <span class="tag no_bad_scan">active</span>' : ''}</td>
          <td>${esc(m.role)}</td><td>${esc(ds)}</td><td>${acc}</td>
          <td>${esc((m.created_at || '').slice(0, 10))}</td>
          <td>${m.is_active ? '' : `<button class="btn sm" data-activate="${m.id}">Activate</button>`}</td>
        </tr>`;
      }).join('') || '<tr><td colspan="6" class="empty">No models yet.</td></tr>';
    } catch (e) { toast(e.message, true); }
  }

  /* ---------------- events ---------------- */

  $('runsTable').addEventListener('click', async (ev) => {
    const tr = ev.target.closest('tbody tr[data-run]');
    if (!tr) return;
    try {
      const d = await get(`/api/training/runs/${tr.dataset.run}/history`);
      if (!d.history || !d.history.length) {
        toast('That run recorded no per-epoch metrics');
        return;
      }
      state.history = d.history;
      $('runpanel').hidden = false;
      $('runSpin').style.visibility = 'hidden';
      $('runTitle').textContent = `${d.model_version || `run #${d.run_id}`} ← ${d.dataset_version || ''}`;
      $('runSub').textContent = `${d.status} — final top-1 `
        + (typeof d.top1_accuracy === 'number' ? `${(d.top1_accuracy * 100).toFixed(1)}%` : 'n/a');
      $('runEpoch').textContent = `${d.history.length} epochs`;
      $('runWhere').textContent = 'history';
      $('runBar').style.width = '100%';
      drawCharts();
      $('runpanel').scrollIntoView({ behavior: 'smooth', block: 'center' });
    } catch (e) { toast(e.message, true); }
  });

  $('modelsTable').addEventListener('click', async (ev) => {
    const btn = ev.target.closest('button[data-activate]');
    if (!btn) return;
    try {
      await post(`/api/training/models/${btn.dataset.activate}/activate`, {});
      toast('Activated');
      loadModels();
      loadState();
    } catch (e) { toast(e.message, true); }
  });

  $('previewBtn').addEventListener('click', async () => {
    try {
      const d = await post('/api/training/preview', { role: state.role });
      const s = d.selection;
      $('selectionPanel').hidden = false;
      if (!s) {
        $('selectionText').textContent = 'This role uses every example in the store.';
        $('selectionChips').innerHTML = '';
        return;
      }
      $('selectionText').textContent =
        `Every bad scan and every past model mistake is kept; clean pages are `
        + `sampled to ${s.neg_sample_ratio}× the positives, spread across projects`
        + `${s.neg_sample_seed === null ? ' (re-drawn each build)' : ` (seed ${s.neg_sample_seed})`}.`;
      const perProj = Object.entries(s.per_project_sampled || {})
        .filter(([, n]) => n > 0)
        .map(([k, n]) => `<span class="chip">project ${esc(k)} <b>${n}</b></span>`).join('');
      $('selectionChips').innerHTML = [
        `<span class="chip bad">bad scans <b>${s.positives}</b></span>`,
        `<span class="chip warn">past mistakes <b>${s.model_errors}</b></span>`,
        `<span class="chip good">clean sampled <b>${s.sampled_negatives}</b></span>`,
        `<span class="chip">clean pool <b>${s.negative_pool}</b></span>`,
        `<span class="chip">total <b>${s.chosen_total}</b> of ${s.store_total}</span>`,
      ].join('') + (perProj ? `<span class="chip">·</span>${perProj}` : '');
    } catch (e) { toast(e.message, true); }
  });

  $('startBtn').addEventListener('click', async () => {
    const epochs = Number($('epochs').value) || 50;
    if (!confirm(`Train a new ${state.role} model for ${epochs} epochs?\n\n`
      + 'A fresh dataset version is built first, then the run goes to the GPU box.')) return;
    try {
      await post('/api/training/start', { role: state.role, epochs });
      $('startBtn').disabled = true;
      toast('Training started');
      openStream();
    } catch (e) { toast(e.message, true); }
  });

  $('roleSel').addEventListener('change', (ev) => {
    state.role = ev.target.value;
    $('selectionPanel').hidden = true;
    loadModels();
  });

  window.addEventListener('beforeunload', () => { if (state.es) state.es.close(); });

  loadState();
  loadRuns();
  loadModels();
})();
