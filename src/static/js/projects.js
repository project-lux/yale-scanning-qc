/* Project list: one card per monitored directory. */

(() => {
  const { get, post, del, toast, esc, $ } = QC;

  async function load() {
    try {
      const { projects } = await get('/api/projects');
      render(projects || []);
    } catch (e) {
      toast(e.message, true);
    }
  }

  function render(projects) {
    $('empty').hidden = projects.length > 0;
    $('cards').innerHTML = projects.map((p) => {
      const s = p.stats || {};
      const last = s.last_batch
        ? `${esc(s.last_batch.label)} · ${esc((s.last_batch.started_at || '').slice(0, 10))}`
        : 'never';
      return `
        <div class="card ${p.enabled ? '' : 'off'}" data-id="${p.id}">
          <h3><a href="/project/${p.id}">${esc(p.name)}</a></h3>
          <div class="path" title="${esc(p.path)}">${esc(p.path)}</div>
          ${p.exists ? '' : '<div class="warnrow">Directory not reachable right now</div>'}
          <div class="nums">
            <span class="chip">scans <b>${s.n_scans || 0}</b></span>
            <span class="chip bad">bad <b>${s.n_bad || 0}</b></span>
            <span class="chip">reviewed <b>${s.n_reviewed || 0}</b></span>
            <span class="chip">examples <b>${s.n_examples || 0}</b></span>
          </div>
          <div class="path">Last pass: ${last}</div>
          <div class="acts">
            <a class="btn sm primary" href="/project/${p.id}">Open</a>
            <a class="btn sm" href="/data?project=${p.id}">Training data</a>
            <button class="btn sm" data-act="rename">Rename</button>
            <button class="btn sm" data-act="toggle">${p.enabled ? 'Disable' : 'Enable'}</button>
            <button class="btn sm danger" data-act="remove">Remove</button>
          </div>
        </div>`;
    }).join('');
  }

  $('cards').addEventListener('click', async (ev) => {
    const btn = ev.target.closest('button[data-act]');
    if (!btn) return;
    const card = btn.closest('.card');
    const id = Number(card.dataset.id);
    const act = btn.dataset.act;

    try {
      if (act === 'rename') {
        const name = prompt('Project name:', card.querySelector('h3').textContent);
        if (!name) return;
        await post(`/api/projects/${id}`, { name });
      } else if (act === 'toggle') {
        const enabling = btn.textContent.trim() === 'Enable';
        await post(`/api/projects/${id}`, { enabled: enabling });
      } else if (act === 'remove') {
        if (!confirm('Stop tracking this directory?\n\nScans, batches and training '
          + 'examples are kept — only the project entry is removed.')) return;
        await del(`/api/projects/${id}`);
      }
      load();
    } catch (e) {
      toast(e.message, true);
    }
  });

  $('addBtn').addEventListener('click', async () => {
    const path = $('newPath').value.trim();
    if (!path) { toast('Enter a directory path', true); return; }
    try {
      await post('/api/projects', { path });
      $('newPath').value = '';
      toast('Project added');
      load();
    } catch (e) {
      toast(e.message, true);
    }
  });

  $('newPath').addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') $('addBtn').click();
  });

  load();
})();
