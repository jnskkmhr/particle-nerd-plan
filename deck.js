(() => {
  'use strict';
  const slides = [...document.querySelectorAll('.slide')];
  const counter = document.getElementById('counter');
  const notesDialog = document.getElementById('notes-dialog');
  const overviewButton = document.getElementById('overview');
  let current = 0;
  let overview = false;

  // The contact derivation stays on slide 29; ordinary slide navigation is unchanged.
  const contactTabs = [...document.querySelectorAll('[data-contact-tab]')];
  let contactStep = 0;
  function showContactStep(index, focus = false) {
    contactStep = Math.max(0, Math.min(contactTabs.length - 1, index));
    contactTabs.forEach((tab, i) => {
      const selected = i === contactStep;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      const panel = document.getElementById(tab.getAttribute('aria-controls'));
      panel.hidden = !selected;
      panel.classList.toggle('is-visible', selected);
    });
    document.getElementById('contact-step-back').disabled = contactStep === 0;
    document.getElementById('contact-step-next').textContent = contactStep === 4 ? 'Inputs ↺' : `Step ${contactStep + 1} →`;
    document.getElementById('contact-step-status').textContent = contactStep === 0
      ? 'Inputs and outputs · select a step to see its equations'
      : `Step ${contactStep} of 4 · ${contactTabs[contactStep].textContent.split(' · ')[1]}`;
    if (focus) contactTabs[contactStep].focus();
  }
  contactTabs.forEach((tab, i) => {
    tab.addEventListener('click', () => showContactStep(i));
    tab.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault(); event.stopPropagation();
      showContactStep(event.key === 'Home' ? 0 : event.key === 'End' ? 4 : (contactStep + (event.key === 'ArrowRight' ? 1 : 4)) % 5, true);
    });
  });
  document.getElementById('contact-step-back').addEventListener('click', () => showContactStep(contactStep - 1));
  document.getElementById('contact-step-next').addEventListener('click', () => showContactStep((contactStep + 1) % 5));

  const replaySelector = document.getElementById('replay-case');
  const replayCases = JSON.parse(document.getElementById('replay-cases').textContent);
  replaySelector.addEventListener('change', () => {
    const selected = replayCases[Number(replaySelector.value)];
    const video = document.getElementById('motion-video');
    video.pause();
    video.poster = selected.poster;
    video.querySelector('source').src = selected.video;
    video.querySelector('a').href = selected.video;
    video.load();
    const poster = document.querySelector('.replay-print-poster');
    poster.src = selected.poster;
    poster.alt = selected.label + ': MPM reference and estimated-force replay with synchronized GRF';
    document.getElementById('replay-initial').textContent = selected.description;
    document.getElementById('replay-metrics').textContent = selected.metrics;
    document.getElementById('replay-profile-link').href = selected.profile;
    document.getElementById('replay-profile-link').textContent = selected.id === 'drop'
      ? 'GRF profile and impact detail' : 'GRF, horizontal force, and torque profiles';
    document.getElementById('replay-video-link').href = selected.video;
  });

  function resize() {
    const scale = Math.max(0.05, Math.min((innerWidth - 24) / 1280, (innerHeight - 92) / 720));
    document.documentElement.style.setProperty('--scale', scale);
  }
  function render() {
    slides.forEach((slide, i) => {
      slide.classList.toggle('active', i === current);
      slide.setAttribute('aria-hidden', String(!overview && i !== current));
      slide.inert = !overview && i !== current;
      slide.tabIndex = overview ? 0 : -1;
    });
    counter.textContent = `${current + 1} / ${slides.length}`;
    document.getElementById('progress-bar').style.width = `${(current + 1) / slides.length * 100}%`;
    document.getElementById('prev').disabled = current === 0;
    document.getElementById('next').disabled = current === slides.length - 1;
    document.getElementById('notes-title').textContent = `${current + 1}. ${slides[current].querySelector('h1,h2').textContent}`;
    document.getElementById('notes-body').textContent = slides[current].querySelector('.notes')?.textContent.trim() || 'No speaker notes.';
  }
  function go(index) {
    current = Math.max(0, Math.min(slides.length - 1, index));
    history.replaceState(null, '', `#${current + 1}`);
    render();
  }
  function readHash() {
    const match = location.hash.match(/^#(?:\/)?(\d+)$/);
    current = match ? Math.max(0, Math.min(slides.length - 1, Number(match[1]) - 1)) : 0;
    render();
  }
  function toggleOverview(force) {
    overview = typeof force === 'boolean' ? force : !overview;
    document.body.classList.toggle('overview-mode', overview);
    overviewButton.setAttribute('aria-pressed', String(overview));
    render();
    if (overview) slides[current].focus();
    else { window.scrollTo(0, 0); slides[current].focus({preventScroll:true}); }
  }
  function toggleNotes() {
    if (notesDialog.open) notesDialog.close();
    else notesDialog.showModal();
  }
  async function toggleFullscreen() {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await document.documentElement.requestFullscreen();
    } catch {
      document.getElementById('status').textContent = 'Full screen is unavailable. Use your browser’s full-screen command.';
    }
  }
  document.getElementById('prev').addEventListener('click', () => go(current - 1));
  document.getElementById('next').addEventListener('click', () => go(current + 1));
  overviewButton.addEventListener('click', () => toggleOverview());
  document.getElementById('notes').addEventListener('click', toggleNotes);
  document.getElementById('close-notes').addEventListener('click', () => notesDialog.close());
  document.getElementById('fullscreen').addEventListener('click', toggleFullscreen);
  document.getElementById('print').addEventListener('click', () => { toggleOverview(false); window.print(); });
  slides.forEach((slide, i) => {
    slide.addEventListener('click', () => { if (overview) { go(i); toggleOverview(false); } });
    slide.addEventListener('keydown', event => {
      if (overview && (event.key === 'Enter' || event.key === ' ')) {
        event.preventDefault(); event.stopPropagation(); go(i); toggleOverview(false);
      }
    });
  });
  document.addEventListener('keydown', event => {
    if (event.ctrlKey || event.metaKey || event.altKey || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) return;
    if (notesDialog.open) return;
    if (event.target.closest('button,a') && (event.key === ' ' || event.key === 'Enter')) return;
    const key = event.key.toLowerCase();
    if (key === 'o') { event.preventDefault(); toggleOverview(); return; }
    if (key === 'escape' && overview) { toggleOverview(false); return; }
    if (key === 'n') { toggleNotes(); return; }
    if (key === 'f') { toggleFullscreen(); return; }
    if (overview) return;
    if (['arrowright','arrowdown','pagedown',' '].includes(key)) { event.preventDefault(); go(current + 1); }
    if (['arrowleft','arrowup','pageup'].includes(key)) { event.preventDefault(); go(current - 1); }
    if (key === 'home') { event.preventDefault(); go(0); }
    if (key === 'end') { event.preventDefault(); go(slides.length - 1); }
  });
  let touchStart = null;
  document.getElementById('deck').addEventListener('touchstart', event => {
    if (!overview && event.touches.length === 1) touchStart = {x:event.touches[0].clientX, y:event.touches[0].clientY};
  }, {passive:true});
  document.getElementById('deck').addEventListener('touchend', event => {
    if (!touchStart || overview) return;
    const dx = event.changedTouches[0].clientX - touchStart.x;
    const dy = event.changedTouches[0].clientY - touchStart.y;
    if (Math.abs(dx) > 60 && Math.abs(dx) > Math.abs(dy) * 1.5) go(current + (dx < 0 ? 1 : -1));
    touchStart = null;
  }, {passive:true});
  addEventListener('resize', resize);
  addEventListener('hashchange', readHash);
  addEventListener('beforeprint', () => toggleOverview(false));
  overviewButton.setAttribute('aria-pressed', 'false');
  resize(); readHash();
})();
