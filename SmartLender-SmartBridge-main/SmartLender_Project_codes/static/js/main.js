/**
 * SmartLender — main.js
 * =====================
 * Handles:
 *   1. Dark / light mode toggle + localStorage persistence
 *   2. Live EMI calculator (index page)
 *   3. Animated stat counters (index page)
 *   4. Submit button loading state
 *   5. Drag-and-drop CSV upload (batch page)
 */

/* ============================================================
   1. Theme toggle
   ============================================================ */
(function () {
  const KEY = 'sl-theme';
  const root = document.documentElement;
  const btn  = document.getElementById('theme-toggle');

  function applyTheme(theme) {
    root.setAttribute('data-theme', theme);
    localStorage.setItem(KEY, theme);
  }

  // Restore saved theme
  const saved = localStorage.getItem(KEY) ||
    (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  applyTheme(saved);

  if (btn) {
    btn.addEventListener('click', () => {
      const current = root.getAttribute('data-theme');
      applyTheme(current === 'dark' ? 'light' : 'dark');
    });
  }
})();

/* ============================================================
   2. Live EMI Calculator
   ============================================================ */
(function () {
  const incomeInput   = document.getElementById('applicant_income');
  const coIncomeInput = document.getElementById('coapplicant_income');
  const loanInput     = document.getElementById('loan_amount');
  const termSelect    = document.getElementById('loan_amount_term');

  const emiValue   = document.getElementById('emi-value');
  const emiTotal   = document.getElementById('emi-total');
  const emiInt     = document.getElementById('emi-interest');
  const dtiLabel   = document.getElementById('dti-label');
  const ringFill   = document.getElementById('dti-ring-fill');

  if (!emiValue) return; // Not on index page

  const RING_CIRC = 2 * Math.PI * 32; // radius = 32
  const RATE_PA   = 8.5;

  function fmt(n) {
    return '$' + n.toLocaleString('en-US', { maximumFractionDigits: 0 });
  }

  function calcEMI(principal, annualRate, months) {
    if (months <= 0 || principal <= 0) return 0;
    const r = annualRate / 100 / 12;
    if (r === 0) return principal / months;
    return principal * r * Math.pow(1 + r, months) / (Math.pow(1 + r, months) - 1);
  }

  function update() {
    const income = (parseFloat(incomeInput.value) || 0) +
                   (parseFloat(coIncomeInput ? coIncomeInput.value : 0) || 0);
    const loanK  = parseFloat(loanInput.value) || 0;
    const term   = parseFloat(termSelect.value) || 360;

    if (loanK <= 0) {
      emiValue.textContent = '—';
      emiTotal.textContent = '—';
      emiInt.textContent   = '—';
      dtiLabel.textContent = '—';
      if (ringFill) {
        ringFill.style.strokeDasharray  = RING_CIRC;
        ringFill.style.strokeDashoffset = RING_CIRC;
        ringFill.classList.remove('ring-danger');
      }
      return;
    }

    const emi      = calcEMI(loanK * 1000, RATE_PA, term);
    const totalPay = emi * term;
    const totalInt = totalPay - loanK * 1000;
    const dti      = income > 0 ? (emi / income) * 100 : null;

    emiValue.textContent = fmt(emi);
    emiTotal.textContent = fmt(totalPay);
    emiInt.textContent   = fmt(totalInt);

    if (dti !== null) {
      dtiLabel.textContent = dti.toFixed(0) + '%';
      const fraction = Math.min(dti / 100, 1);
      if (ringFill) {
        ringFill.style.strokeDasharray  = RING_CIRC;
        ringFill.style.strokeDashoffset = RING_CIRC * (1 - fraction);
        ringFill.classList.toggle('ring-danger', dti > 43);
      }
    } else {
      dtiLabel.textContent = '—';
    }
  }

  [incomeInput, coIncomeInput, loanInput, termSelect].forEach(el => {
    if (el) el.addEventListener('input', update);
  });

  update();
})();

/* ============================================================
   3. Animated stat counters
   ============================================================ */
(function () {
  function animateCounter(el, target, duration) {
    target = parseFloat(target) || 0;
    if (target === 0) { el.textContent = '0'; return; }
    let start = null;
    const step = (ts) => {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      el.textContent = (target % 1 === 0)
        ? Math.round(target * eased)
        : (target * eased).toFixed(1);
      if (progress < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  const counters = document.querySelectorAll('.stat-num[data-target]');
  if (!counters.length) return;

  // Use IntersectionObserver for trigger-on-scroll
  const obs = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el = entry.target;
        animateCounter(el, el.dataset.target, 1200);
        obs.unobserve(el);
      }
    });
  }, { threshold: 0.5 });

  counters.forEach(el => obs.observe(el));
})();

/* ============================================================
   4. Form submit → loading spinner
   ============================================================ */
(function () {
  const form = document.getElementById('application-form');
  const btn  = document.getElementById('submit-btn');

  if (!form || !btn) return;

  form.addEventListener('submit', () => {
    btn.classList.add('loading');
    btn.disabled = true;
  });
})();

/* ============================================================
   5. Drag-and-drop CSV upload (batch page)
   ============================================================ */
(function () {
  const zone    = document.getElementById('drop-zone');
  const input   = document.getElementById('csv-file-input');
  const nameEl  = document.getElementById('drop-filename');
  const submitBtn = document.getElementById('batch-submit');

  if (!zone || !input) return;

  function setFile(file) {
    if (!file) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    if (nameEl) nameEl.textContent = file.name;
    zone.classList.add('file-selected');
  }

  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('drag-over');
  });

  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));

  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.csv')) {
      setFile(file);
    } else {
      alert('Please drop a .csv file.');
    }
  });

  input.addEventListener('change', () => {
    const file = input.files[0];
    if (file && nameEl) nameEl.textContent = file.name;
    zone.classList.add('file-selected');
  });

  const batchForm = document.getElementById('batch-form');
  if (batchForm && submitBtn) {
    batchForm.addEventListener('submit', () => {
      submitBtn.classList.add('loading');
      submitBtn.disabled = true;
    });
  }
})();

/* ============================================================
   6. Highlight current nav link
   ============================================================ */
(function () {
  const path = window.location.pathname;
  const navMap = {
    '/':         'nav-application',
    '/batch':    'nav-batch',
    '/history':  'nav-history',
    '/reports':  'nav-reports',
    '/about':    'nav-about',
  };

  const id = navMap[path];
  if (id) {
    const el = document.getElementById(id);
    if (el) el.style.color = '#fff';
  }
})();
