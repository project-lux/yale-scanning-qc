/* Shared helpers: fetch wrappers, toasts, small DOM utilities. */

const QC = (() => {
  async function req(url, opts = {}) {
    const res = await fetch(url, {
      headers: { 'Content-Type': 'application/json' },
      ...opts,
    });
    let body = null;
    try { body = await res.json(); } catch { /* empty or non-JSON */ }
    if (!res.ok) {
      const msg = (body && body.error) || `${res.status} ${res.statusText}`;
      throw new Error(msg);
    }
    return body;
  }

  const get = (url) => req(url);
  const post = (url, data) => req(url, { method: 'POST', body: JSON.stringify(data || {}) });
  const del = (url) => req(url, { method: 'DELETE' });

  let toastTimer = null;
  function toast(msg, isError = false) {
    const el = document.getElementById('toast');
    if (!el) return;
    el.textContent = msg;
    el.classList.toggle('err', !!isError);
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.hidden = true; }, isError ? 6000 : 2600);
  }

  /* Escape text destined for innerHTML. */
  function esc(s) {
    return String(s ?? '').replace(/[&<>"']/g, (c) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
  }

  const $ = (id) => document.getElementById(id);

  /* Human-friendly class name: bad_scan -> "bad scan" */
  const pretty = (s) => String(s ?? '').replace(/_/g, ' ');

  return { get, post, del, toast, esc, $, pretty };
})();
