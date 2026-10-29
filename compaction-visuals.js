(() => {
  document.querySelectorAll('[data-cp-tab]').forEach(button => {
    button.addEventListener('click', () => {
      const slide = button.closest('.cp-slide');
      slide.querySelectorAll('[data-cp-tab]').forEach(tab => tab.setAttribute('aria-pressed', String(tab === button)));
      slide.querySelectorAll('[data-cp-panel]').forEach(panel => { panel.hidden = panel.dataset.cpPanel !== button.dataset.cpTab; });
    });
  });
  document.querySelectorAll('.cp-pause').forEach(button => {
    button.addEventListener('click', () => {
      const slide = button.closest('.cp-slide');
      const paused = slide.classList.toggle('cp-paused');
      slide.querySelectorAll('.cp-pause').forEach(control => { control.textContent = paused ? 'Play animation' : 'Pause animation'; control.setAttribute('aria-pressed', String(paused)); });
    });
  });
})();
