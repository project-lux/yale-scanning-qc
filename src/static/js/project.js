/* Project detail: scan grid, plain-language review, run a pass, reprocess. */

(() => {
  const { get, post, toast, esc, $, pretty } = QC;

  const root = $('projectRoot');
  const PID = Number(root.dataset.projectId);
  const TREE = root.dataset.mode === 'tree';
  const PAGE_TYPES = (root.dataset.pageTypes || '').split(',').filter(Boolean);

  const state = {
    page: 1,
    perPage: 60,
    rows: [],
    selected: new Set(),
    openIdx: -1,
    polling: null,
    similar: null,      // {id, filename} while ranking by likeness to one scan
    minScore: 0,
    view: localStorage.getItem('qc-project-view') || 'grid',
  };

  /* ---------------- loading ---------------- */

  function query() {
    const p = new URLSearchParams({ page: String(state.page), per_page: String(state.perPage) });
    if ($('fFlagged').value) p.set('flagged', $('fFlagged').value);
    if ($('fReviewed').value === 'kept') p.set('kept', '1');
    else if ($('fReviewed').value) p.set('reviewed', $('fReviewed').value);
    // Always sent: the server's default is "on disk", and '' means all.
    p.set('present', $('fPresent').value);
    if ($('fPageType').value) p.set('page_type', $('fPageType').value);
    if ($('fTranscribed').value) p.set('transcribed', $('fTranscribed').value);
    if (TREE && $('fDir').value) p.set('dir', $('fDir').value);
    if (state.similar) {
      p.set('similar_to', String(state.similar.id));
      if (state.minScore) p.set('min_score', String(state.minScore));
    }
    return p.toString();
  }

  async function load() {
    try {
      const d = await get(`/api/projects/${PID}/scans?${query()}`);
      state.rows = d.scans || [];
      renderCounts(d.counts || {});
      renderSimbar(d.similarity);
      render();
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
      `<span class="chip kept" title="Scans kept as training data (the model got them wrong)">kept <b>${c.kept || 0}</b></span>`,
      c.gone ? `<span class="chip warn" title="Seen on an earlier pass, no longer in the folder">gone <b>${c.gone}</b></span>` : '',
      c.errors ? `<span class="chip warn">errors <b>${c.errors}</b></span>` : '',
      c.transcribed ? `<span class="chip">transcribed <b>${c.transcribed}</b></span>` : '',
    ].filter(Boolean).join('');
  }

  /* ---------------- similar-image mode ---------------- */

  function renderSimbar(st) {
    $('simbar').hidden = !state.similar;
    if (!state.similar) return;
    $('simName').textContent = state.similar.filename;
    $('simThumb').src = `/api/projects/scans/${state.similar.id}/image`;
    let msg = `${state.rows.length} shown, best match first`;
    if (st && st.running) {
      msg += ` · indexing ${st.done}/${st.todo} new images — results fill in as it runs`;
    }
    $('simStatus').textContent = msg;
  }

  function findSimilar(r) {
    if (!r) return;
    state.similar = { id: r.id, filename: r.relpath || r.filename };
    state.page = 1;
    closeDrawer();
    window.scrollTo({ top: 0 });
    load();
  }

  function exitSimilar() {
    state.similar = null;
    state.page = 1;
    load();
  }

  $('simExit').addEventListener('click', exitSimilar);
  $('simSelect').addEventListener('click', () => {
    state.rows.forEach((r) => state.selected.add(r.id));
    render();
  });
  $('simMin').addEventListener('input', (ev) => {
    const v = Number(ev.target.value);
    $('simMinVal').textContent = v ? `${v}%` : 'any';
  });
  $('simMin').addEventListener('change', (ev) => {
    state.minScore = Number(ev.target.value) / 100;
    state.page = 1;
    load();
  });

  /* A review is stored as TP/FP/FN/TN; show it as what the human decided. */
  function reviewLabel(v) {
    if (v === 'true_positive' || v === 'false_negative') return 'bad';
    if (v === 'false_positive' || v === 'true_negative') return 'fine';
    return '';
  }

  /* A gone file only has a preview if a copy was kept (has_jpeg covers that). */
  function hasPreview(r) {
    return r.has_jpeg || (r.present !== 0 && /\.(jpe?g|png)$/i.test(r.filepath));
  }

  function badgesFor(r) {
    const verdict = r.bad_scan
      ? `<span class="tag ${esc(r.bad_scan)}">${esc(pretty(r.bad_scan))}` +
        `${r.bad_scan_conf ? ` ${(r.bad_scan_conf * 100).toFixed(0)}%` : ''}</span>`
      : '<span class="tag">unscored</span>';
    const sim = r.similarity != null
      ? `<span class="tag sim" title="Visual similarity">${(r.similarity * 100).toFixed(0)}%</span>` : '';
    const kept = r.kept
      ? '<span class="tag kept" title="Kept as training data — the model got this one wrong">kept</span>' : '';
    const gone = r.present === 0
      ? '<span class="tag gone" title="No longer in the folder">gone</span>' : '';
    return sim + verdict + pageTypeBadge(r) + kept + gone;
  }

  const isQuery = (r) => state.similar && state.similar.id === r.id;

  /* A reviewer's page type is a label; a model/LLM page type is only a
     guess until someone confirms it, so it is shown as one. */
  function pageTypeBadge(r) {
    if (r.review_page_type) {
      return `<span class="tag rev" title="Your page type">${esc(pretty(r.review_page_type))}</span>`;
    }
    if (!r.page_type) return '';
    const src = (r.page_type_source || '').startsWith('llm') ? 'LLM' : 'model';
    const conf = r.page_type_conf != null && src === 'model'
      ? ` ${(r.page_type_conf * 100).toFixed(0)}%` : '';
    return `<span class="tag guess" title="${src} guess — not reviewed">` +
      `${esc(pretty(r.page_type))}?${conf}</span>`;
  }

  function render() {
    const isList = state.view === 'list';
    $('grid').hidden = isList;
    $('tlist').hidden = !isList;
    $('empty').hidden = state.rows.length > 0;
    if (isList) renderList(); else renderGrid();
    updateBulk();
  }

  function renderGrid() {
    $('tlist').innerHTML = '';
    $('grid').innerHTML = state.rows.map((r, i) => {
      const img = hasPreview(r)
        ? `<img class="thumb" loading="lazy" src="/api/projects/scans/${r.id}/image" alt="">`
        : '<div class="missing">no preview</div>';
      const rev = reviewLabel(r.review_verdict);
      return `
        <div class="tile ${state.selected.has(r.id) ? 'sel' : ''} ${isQuery(r) ? 'query' : ''}"
             data-idx="${i}" data-id="${r.id}">
          <input type="checkbox" class="cb" ${state.selected.has(r.id) ? 'checked' : ''}
                 aria-label="Select ${esc(r.filename)}">
          ${rev ? `<span class="flag">${esc(rev)}</span>` : ''}
          <button class="simbtn" title="Show images that look like this one">≈ Similar</button>
          ${img}
          <div class="tmeta">
            <div class="badges">${badgesFor(r)}</div>
            <div class="fname" title="${esc(r.filepath)}">${esc(r.relpath || r.filename)}</div>
          </div>
        </div>`;
    }).join('');
  }

  /* Transcription view: page image beside its OCR/LLM text and verdicts,
     so the text can be read against the scan it came from. */
  function renderList() {
    $('grid').innerHTML = '';
    $('tlist').innerHTML = state.rows.map((r, i) => {
      const img = hasPreview(r)
        ? `<img class="thumb" loading="lazy" src="/api/projects/scans/${r.id}/image" alt="">`
        : '<div class="missing">no preview</div>';
      const rev = reviewLabel(r.review_verdict);
      const text = r.transcription
        ? `<pre class="ttext">${esc(r.transcription)}</pre>`
        : '<p class="hint">No transcription for this page.</p>';
      return `
        <div class="trow ${state.selected.has(r.id) ? 'sel' : ''} ${isQuery(r) ? 'query' : ''}"
             data-idx="${i}" data-id="${r.id}">
          <div class="timg">
            <input type="checkbox" class="cb" ${state.selected.has(r.id) ? 'checked' : ''}
                   aria-label="Select ${esc(r.filename)}">
            ${img}
          </div>
          <div class="tbody">
            <div class="trowhead">
              <span class="fname" title="${esc(r.filepath)}">${esc(r.relpath || r.filename)}</span>
              <span class="badges">
                ${badgesFor(r)}
                ${rev ? `<span class="tag rev">reviewed: ${esc(rev)}</span>` : ''}
                ${r.transcription_backend ? `<span class="tag src">${esc(r.transcription_backend)}</span>` : ''}
              </span>
            </div>
            ${text}
          </div>
        </div>`;
    }).join('');
  }

  function updateBulk() {
    const n = state.selected.size;
    $('bulkbar').hidden = n === 0;
    $('selCount').textContent = String(n);
    $('reprocessScope').textContent = n ? `${n} selected scan(s)` : 'whole project';
  }

  /* ---------------- viewer + review ---------------- */

  const SVG_NS = 'http://www.w3.org/2000/svg';

  function parseGeometry(raw) {
    if (!raw) return null;
    try {
      const g = JSON.parse(raw);
      return g && g.w && g.h && Array.isArray(g.lines) && g.lines.length ? g : null;
    } catch { return null; }
  }

  /* Line polygons over the page image; hovering either side highlights both. */
  function renderOverlay(geom) {
    const svg = $('dOverlay');
    svg.innerHTML = '';
    if (!geom) { svg.style.display = 'none'; return; }
    svg.style.display = '';
    svg.setAttribute('viewBox', `0 0 ${geom.w} ${geom.h}`);
    geom.lines.forEach((ln, i) => {
      const pts = ln.polygon && ln.polygon.length >= 3 ? ln.polygon
        : (ln.baseline || []);
      if (pts.length < 2) return;
      const poly = document.createElementNS(SVG_NS, 'polygon');
      poly.setAttribute('points', pts.map((p) => p.join(',')).join(' '));
      poly.dataset.line = String(i);
      if (ln.text) {
        const title = document.createElementNS(SVG_NS, 'title');
        title.textContent = ln.text;
        poly.appendChild(title);
      }
      svg.appendChild(poly);
    });
  }

  /* Show one backend's transcription (text + polygons when it has them). */
  function showTranscription(entry) {
    const box = $('dTranscript');
    const out = $('dTranscriptText');
    const geom = entry ? entry.geom : null;
    renderOverlay(geom);
    $('dTranscriptBackend').textContent = entry ? (entry.backend || '') : '';
    $('dLineHint').textContent = geom
      ? `${geom.lines.length} line(s) — hover a box or a text row`
      : '';
    if (geom) {
      box.hidden = false;
      out.innerHTML = geom.lines.map((ln, i) => {
        const conf = ln.confidence != null ? ` · ${(ln.confidence * 100).toFixed(0)}%` : '';
        return `<span class="tline" data-line="${i}" title="line ${i + 1}${conf}">${esc(ln.text)}</span>`;
      }).join('');
    } else if (entry && entry.text) {
      box.hidden = false;
      out.innerHTML = `<pre>${esc(entry.text)}</pre>`;
    } else if (entry) {
      box.hidden = false;
      out.innerHTML = '<pre class="hint">(no text found on this page)</pre>';
    } else {
      box.hidden = true;
      out.innerHTML = '';
    }
  }

  /* Swap tabs — one per backend that has transcribed this scan. */
  function applyTranscriptions(r) {
    const entries = r._trans || [];
    const tabs = $('dTransTabs');
    if (!entries.length) {
      tabs.innerHTML = '';
      return;
    }
    const active = entries.find((e) => e.backend === r._activeBackend)
      || entries.find((e) => e.backend === r.transcription_backend)
      || entries[0];
    r._activeBackend = active.backend;
    showTranscription(active);
    tabs.innerHTML = entries.length < 2 ? '' : entries.map((e) =>
      `<button class="ttab ${e.backend === active.backend ? 'on' : ''}"
               data-backend="${esc(e.backend)}">${esc(e.backend)}</button>`).join('');
  }

  $('dTransTabs').addEventListener('click', (ev) => {
    const b = ev.target.closest('.ttab');
    const r = state.rows[state.openIdx];
    if (!b || !r) return;
    r._activeBackend = b.dataset.backend;
    applyTranscriptions(r);
  });

  function setLineHl(i, on) {
    document.querySelectorAll(`#dOverlay polygon[data-line="${i}"], #dTranscriptText .tline[data-line="${i}"]`)
      .forEach((el) => el.classList.toggle('hl', on));
  }

  function bindHover(container, scrollTarget) {
    let last = null;
    container.addEventListener('mouseover', (ev) => {
      const t = ev.target.closest('[data-line]');
      const i = t ? t.dataset.line : null;
      if (i === last) return;
      if (last !== null) setLineHl(last, false);
      last = i;
      if (i === null) return;
      setLineHl(i, true);
      const other = document.querySelector(`${scrollTarget}[data-line="${i}"]`);
      if (other && other.scrollIntoView) other.scrollIntoView({ block: 'nearest' });
    });
    container.addEventListener('mouseleave', () => {
      if (last !== null) setLineHl(last, false);
      last = null;
    });
  }
  bindHover($('dOverlay'), '#dTranscriptText .tline');
  bindHover($('dTranscriptText'), '#dOverlay polygon');

  function openDrawer(idx) {
    const r = state.rows[idx];
    if (!r) return;
    state.openIdx = idx;

    $('dFilename').textContent = r.filename;
    $('dFilename').title = r.filepath;
    const img = $('dImg');
    $('vImgWrap').classList.remove('actual');
    if (hasPreview(r)) {
      img.src = `/api/projects/scans/${r.id}/image?full=1`;
      img.style.display = '';
    } else {
      img.removeAttribute('src');
      img.style.display = 'none';
    }

    const rows = [
      ['Model says', r.bad_scan ? `${pretty(r.bad_scan)}` +
        (r.bad_scan_conf ? ` (${(r.bad_scan_conf * 100).toFixed(1)}%)` : '') : 'not scored'],
      ['Model page type', r.page_type ? `${pretty(r.page_type)}${r.page_type_source ? ` · ${r.page_type_source}` : ''}` : '—'],
      ['Your review', (reviewLabel(r.review_verdict) || 'not reviewed') +
        (r.reviewer ? ` · ${r.reviewer}` : '') +
        (r.reviewed_at ? ` · ${r.reviewed_at.slice(0, 16).replace('T', ' ')}` : '')],
      ['Your page type', r.review_page_type ? pretty(r.review_page_type) : '—'],
      ['Kept', r.kept ? 'yes — copy and metadata are in the training data' : 'no'],
      ['File', r.present === 0 ? 'gone from the folder' : 'on disk'],
      ['Pass', `${r.batch_label || '—'} · ${(r.batch_started || '').slice(0, 10)}`],
      ['Status', r.status + (r.error ? ` — ${r.error}` : '')],
      ['Path', r.filepath],
    ];
    $('dMeta').innerHTML = rows.map(([k, v]) =>
      `<div class="row"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`
    ).join('');

    // Show the pass's own transcription immediately, then fetch every
    // backend's reading for the swap tabs.
    showTranscription(r.transcription != null ? {
      backend: r.transcription_backend || '',
      text: r.transcription,
      geom: parseGeometry(r.transcription_lines),
    } : null);
    $('dTransTabs').innerHTML = '';
    if (r._trans) {
      applyTranscriptions(r);
    } else {
      get(`/api/projects/scans/${r.id}/transcriptions`).then((d) => {
        r._trans = (d.transcriptions || []).map((t) => ({
          backend: t.backend,
          text: t.text,
          geom: parseGeometry(t.lines),
        }));
        if (state.rows[state.openIdx] === r) applyTranscriptions(r);
      }).catch(() => {});
    }

    // Show which bookkeeping verdict the answer will record, and what
    // happened to the image as a result.
    const flagged = r.bad_scan === 'bad_scan';
    const cur = reviewLabel(r.review_verdict);
    let hint = flagged ? 'The model flagged this one.' : 'The model did not flag this one.';
    if (cur) {
      const wrong = r.review_verdict === 'false_positive' || r.review_verdict === 'false_negative';
      hint += wrong
        ? ' You overruled it — a copy is kept as training data.'
        : ' You agreed — nothing kept.';
    }
    $('verdictHint').textContent = hint;
    $('vBad').classList.toggle('on', cur === 'bad');
    $('vGood').classList.toggle('on', cur === 'fine');

    $('dPageType').value = r.review_page_type || '';
    $('dNotes').value = r.review_notes || '';
    loadTrainLabels(r);
    loadHistory(r);
    $('drawer').hidden = false;
  }

  /* Everything an annotator has done to this scan, newest first. */
  async function loadHistory(r) {
    const box = $('dHistory');
    box.innerHTML = '';
    let rows;
    try {
      rows = (await get(`/api/projects/scans/${r.id}/history`)).history || [];
    } catch { return; }
    if (state.rows[state.openIdx] !== r) return;
    box.innerHTML = '<div class="lhead">History</div>' +
      (rows.length ? rows.map(historyRow).join('') : '<p class="hint">No annotations yet.</p>');
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

  /* The training-data rows built from this scan: what each model will learn
     from it now, and what it was first imported as. */
  async function loadTrainLabels(r) {
    const box = $('dTrainLabels');
    box.innerHTML = '';
    let rows;
    try {
      rows = (await get(`/api/projects/scans/${r.id}/labels`)).examples || [];
    } catch { return; }
    if (state.rows[state.openIdx] !== r) return;
    if (!rows.length) {
      box.innerHTML = '<div class="lhead">Training labels</div>' +
        '<p class="hint">Not in the training data yet.</p>';
      return;
    }
    box.innerHTML = '<div class="lhead">Training labels</div>' + rows.map((e) => {
      const changed = e.original_label && e.original_label !== e.label;
      return `<div class="lrow">
        <span class="lrole">${esc(e.role === 'badscan' ? 'Bad scan' : 'Page type')}</span>
        <span class="tag ${esc(e.label)}">${esc(pretty(e.label))}</span>
        ${changed ? `<span class="hint">was ${esc(pretty(e.original_label))}</span>` : ''}
        <span class="tag src">${esc(e.source)}</span>
        ${e.active ? '' : '<span class="tag">excluded</span>'}
      </div>`;
    }).join('');
  }

  async function savePageType() {
    if (state.openIdx < 0) return;
    const r = state.rows[state.openIdx];
    const pt = $('dPageType').value || null;
    // Give focus back to the page so ←/→ move between scans, not options.
    $('dPageType').blur();
    try {
      await post(`/api/projects/scans/${r.id}/page-type`, { page_type: pt });
      r.review_page_type = pt;
      toast(pt ? `Page type set to ${pretty(pt)}` : 'Page type cleared');
      render();
      if (state.rows[state.openIdx] === r) openDrawer(state.openIdx);
    } catch (e) {
      toast(e.message, true);
    }
  }

  const closeDrawer = () => { $('drawer').hidden = true; state.openIdx = -1; };
  // Click on the dimmed backdrop (outside the box) closes the viewer.
  $('drawer').addEventListener('click', (ev) => {
    if (ev.target === $('drawer')) closeDrawer();
  });

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
        has_jpeg: r.has_jpeg || !!d.kept,
        kept: !!d.kept,
      });
      const outcome = d.disagreement
        ? 'model was wrong — kept for training'
        : 'model agreed — not kept';
      toast(`Recorded: ${isBad ? 'bad scan' : 'looks fine'} · ${outcome}`);
      render();
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
          loadDirs();
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

  /* Subdirectory filter for tree projects: pick a branch or stay at the
     top ("All directories"). Reloaded after every pass/rescan so new
     folders show up. */
  async function loadDirs() {
    if (!TREE) return;
    try {
      const d = await get(`/api/projects/${PID}/dirs`);
      const dirs = (d.dirs || []).filter((x) => x.dir);
      const cur = $('fDir').value;
      $('fDir').innerHTML = '<option value="">All directories</option>' + dirs.map((x) => {
        const depth = x.dir.split('/').length - 1;
        const label = `${'  '.repeat(depth)}${x.dir.split('/').pop()} (${x.n})`;
        return `<option value="${esc(x.dir)}">${label}</option>`;
      }).join('');
      if ([...$('fDir').options].some((o) => o.value === cur)) $('fDir').value = cur;
      $('dirFilterWrap').hidden = false;
    } catch { /* filter stays hidden */ }
  }

  const rescanBtn = document.getElementById('rescanBtn');
  if (rescanBtn) {
    rescanBtn.addEventListener('click', async () => {
      try {
        await post(`/api/projects/${PID}/run`, { all: false });
        setRunning(true, 'Rescanning directory', 'looking for new images…', 0);
        startPolling();
      } catch (e) { toast(e.message, true); }
    });
  }

  /* Bulk review: same plain answer applied to every selected scan. */
  async function bulkVerdict(isBad) {
    const ids = [...state.selected];
    if (!ids.length) { toast('Select some scans first', true); return; }
    const what = isBad ? 'bad scans' : 'fine';
    if (!confirm(`Mark ${ids.length} selected scan(s) as ${what}?\n\n`
      + 'Each becomes a review; the ones the model got wrong are kept as training data.')) return;
    try {
      const d = await post(`/api/projects/${PID}/bulk-verdict`, {
        scan_ids: ids,
        is_bad: isBad,
        notes: $('bulkNotes').value.trim(),
      });
      toast(`Reviewed ${d.reviewed} scan(s) as ${what} · ${d.kept || 0} kept for training`);
      state.selected.clear();
      $('bulkNotes').value = '';
      load();
    } catch (e) { toast(e.message, true); }
  }
  /* Bulk page type: applies as soon as one is picked. */
  $('bulkPageType').addEventListener('change', async (ev) => {
    const pt = ev.target.value;
    const ids = [...state.selected];
    ev.target.blur();
    if (!pt || !ids.length) return;
    if (!confirm(`Set ${ids.length} selected scan(s) to "${pretty(pt)}"?`)) {
      ev.target.value = '';
      return;
    }
    try {
      const d = await post(`/api/projects/${PID}/bulk-page-type`, { scan_ids: ids, page_type: pt });
      toast(`Set ${d.updated} scan(s) to ${pretty(pt)}`);
      state.selected.clear();
      ev.target.value = '';
      load();
    } catch (e) { toast(e.message, true); }
  });

  $('bulkBad').addEventListener('click', () => bulkVerdict(true));
  $('bulkGood').addEventListener('click', () => bulkVerdict(false));

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
      if ($('stForce').checked) body.force_transcribe = true;
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

  function onRowClick(ev) {
    const tile = ev.target.closest('.tile, .trow');
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
      updateBulk();
      return;
    }
    openDrawer(Number(tile.dataset.idx));
  }
  $('grid').addEventListener('click', onRowClick);
  $('tlist').addEventListener('click', onRowClick);

  $('vBad').addEventListener('click', () => submitVerdict(true));
  $('vGood').addEventListener('click', () => submitVerdict(false));
  $('dClose').addEventListener('click', closeDrawer);
  $('dPageType').addEventListener('change', savePageType);
  $('dSimilar').addEventListener('click', () => findSimilar(state.rows[state.openIdx]));
  $('vImgWrap').addEventListener('click', () => $('vImgWrap').classList.toggle('actual'));
  $('dPrev').addEventListener('click', () => openDrawer(Math.max(0, state.openIdx - 1)));
  $('dNext').addEventListener('click', () =>
    openDrawer(Math.min(state.rows.length - 1, state.openIdx + 1)));

  $('selAll').addEventListener('click', () => {
    state.rows.forEach((r) => state.selected.add(r.id));
    render();
  });
  $('selNone').addEventListener('click', () => { state.selected.clear(); render(); });

  ['fFlagged', 'fReviewed', 'fPageType', 'fTranscribed', 'fDir', 'fPresent'].forEach((id) =>
    $(id).addEventListener('change', () => { state.page = 1; load(); }));
  $('fView').value = state.view;
  $('fView').addEventListener('change', (ev) => {
    state.view = ev.target.value === 'list' ? 'list' : 'grid';
    localStorage.setItem('qc-project-view', state.view);
    render();
  });
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
    if (ev.key === 'Escape') {
      if (state.openIdx >= 0) closeDrawer();
      else if (state.similar) exitSimilar();
      return;
    }
    if (state.openIdx < 0) return;
    if (ev.key === 's') findSimilar(state.rows[state.openIdx]);
    else if (ev.key === 'b') submitVerdict(true);
    else if (ev.key === 'g') submitVerdict(false);
    else if (ev.key === 'ArrowRight' || ev.key === 'j') {
      ev.preventDefault();
      openDrawer(Math.min(state.rows.length - 1, state.openIdx + 1));
    } else if (ev.key === 'ArrowLeft' || ev.key === 'k') {
      ev.preventDefault();
      openDrawer(Math.max(0, state.openIdx - 1));
    }
  });

  /* page-type pickers */
  const opts = PAGE_TYPES.map((t) => `<option value="${esc(t)}">${esc(pretty(t))}</option>`).join('');
  $('fPageType').innerHTML = '<option value="">All</option>' + opts;
  $('dPageType').innerHTML = '<option value="">— unchanged —</option>' + opts;
  $('bulkPageType').innerHTML = '<option value="">— unchanged —</option>' + opts;

  load();
  loadDirs();
  // A pass may already be running when the page opens.
  get(`/api/projects/${PID}/progress`).then((d) => { if (d.running) startPolling(); }).catch(() => {});
})();
