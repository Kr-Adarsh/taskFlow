(() => {
  const key = 'taskflow.theme';
  const valid = value => value === 'dark' || value === 'light';
  const read = () => {
    try { return localStorage.getItem(key); } catch { return null; }
  };
  const set = (theme, persist = true) => {
    if (!valid(theme)) return;
    document.documentElement.dataset.theme = theme;
    if (persist) {
      try { localStorage.setItem(key, theme); } catch {}
    }
  };
  set(valid(read()) ? read() : (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'), false);
  window.TaskFlowTheme = {get: () => document.documentElement.dataset.theme, set};
  addEventListener('storage', event => {
    if (event.key === key && valid(event.newValue)) set(event.newValue, false);
  });
})();
