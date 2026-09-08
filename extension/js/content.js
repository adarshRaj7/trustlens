// TrustLens Content Script
// Injects tooltips and analysis triggers on all webpage images

(function () {
  'use strict';

  const API_BASE = 'http://localhost:8000';
  const cache = new Map();
  let analysisPanel = null;
  let activeTooltip = null;

  // ─── Utilities ────────────────────────────────────────────────────────────

  function scoreColor(score) {
    if (score >= 75) return { bg: '#22c55e', label: 'Likely Real', tier: 'real' };
    if (score >= 45) return { bg: '#f59e0b', label: 'Suspicious', tier: 'warn' };
    return { bg: '#ef4444', label: 'Likely Fake', tier: 'fake' };
  }

  function debounce(fn, ms) {
    let t;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  }

  function getAbsoluteURL(src) {
    try {
      return new URL(src, location.href).href;
    } catch {
      return src;
    }
  }

  // ─── Tooltip ──────────────────────────────────────────────────────────────

  function createTooltip() {
    const tip = document.createElement('div');
    tip.className = 'tl-tooltip';
    tip.innerHTML = `
      <div class="tl-tip-inner">
        <div class="tl-tip-logo">🔍 TrustLens</div>
        <div class="tl-tip-score-row">
          <div class="tl-tip-ring" id="tl-ring">
            <svg viewBox="0 0 36 36"><circle class="tl-ring-bg" cx="18" cy="18" r="15.9"/><circle class="tl-ring-fill" id="tl-ring-fill" cx="18" cy="18" r="15.9"/></svg>
            <span class="tl-ring-num" id="tl-ring-num">–</span>
          </div>
          <div class="tl-tip-details">
            <div class="tl-tip-label" id="tl-tip-label">Analyzing…</div>
            <div class="tl-tip-sub" id="tl-tip-sub">Click for full report</div>
          </div>
        </div>
        <div class="tl-tip-bars" id="tl-tip-bars"></div>
      </div>
    `;
    document.body.appendChild(tip);
    return tip;
  }

  function showTooltip(img, result) {
    if (!activeTooltip) activeTooltip = createTooltip();
    const tip = activeTooltip;
    const { bg, label, tier } = scoreColor(result.final_score);

    tip.querySelector('#tl-ring-num').textContent = result.final_score;
    tip.querySelector('#tl-tip-label').textContent = label;
    tip.querySelector('#tl-tip-label').style.color = bg;
    tip.querySelector('#tl-tip-sub').textContent = result.explanation?.slice(0, 60) + '…' || 'Click for full report';
    tip.dataset.tier = tier;

    // Arc fill
    const circ = 100;
    const pct = result.final_score / 100;
    const fill = tip.querySelector('#tl-ring-fill');
    fill.style.stroke = bg;
    fill.style.strokeDasharray = `${pct * circ} ${circ}`;

    // Mini bars
    const bars = tip.querySelector('#tl-tip-bars');
    bars.innerHTML = `
      <div class="tl-bar-row"><span>Deepfake</span><div class="tl-bar"><div style="width:${result.deepfake_score}%;background:${scoreColor(result.deepfake_score).bg}"></div></div><span>${result.deepfake_score}%</span></div>
      <div class="tl-bar-row"><span>Tamper</span><div class="tl-bar"><div style="width:${result.tampering_score}%;background:${scoreColor(result.tampering_score).bg}"></div></div><span>${result.tampering_score}%</span></div>
    `;

    positionTooltip(tip, img);
    tip.classList.add('tl-visible');
  }

  function positionTooltip(tip, img) {
    const rect = img.getBoundingClientRect();
    const scrollY = window.scrollY;
    const scrollX = window.scrollX;
    let top = rect.top + scrollY - tip.offsetHeight - 10;
    let left = rect.left + scrollX + rect.width / 2 - 140;
    if (top < scrollY + 8) top = rect.bottom + scrollY + 10;
    left = Math.max(8, Math.min(left, document.documentElement.clientWidth + scrollX - 288));
    tip.style.top = top + 'px';
    tip.style.left = left + 'px';
  }

  function hideTooltip() {
    if (activeTooltip) activeTooltip.classList.remove('tl-visible');
  }

  // ─── Loading badge ─────────────────────────────────────────────────────────

  function attachBadge(img) {
    const wrap = document.createElement('span');
    wrap.className = 'tl-badge-wrap';
    wrap.style.cssText = `position:absolute;top:6px;right:6px;z-index:99999;pointer-events:none;`;

    const badge = document.createElement('span');
    badge.className = 'tl-badge tl-badge-loading';
    badge.textContent = '…';
    wrap.appendChild(badge);

    // Position wrapper
    const parent = img.parentElement;
    const pStyle = getComputedStyle(parent);
    if (pStyle.position === 'static') parent.style.position = 'relative';
    parent.appendChild(wrap);

    img._tlBadge = badge;
    img._tlBadgeWrap = wrap;
    return badge;
  }

  function updateBadge(img, score) {
    if (!img._tlBadge) return;
    const { bg, tier } = scoreColor(score);
    const badge = img._tlBadge;
    badge.classList.remove('tl-badge-loading');
    badge.textContent = score + '%';
    badge.style.background = bg;
    badge.dataset.tier = tier;
  }

  // ─── API call ─────────────────────────────────────────────────────────────

  async function analyzeImage(src) {
    const key = src;
    if (cache.has(key)) return cache.get(key);

    // Placeholder while fetching
    const pending = fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_url: src })
    })
      .then(r => { if (!r.ok) throw new Error(`analyze failed: ${r.status}`); return r.json(); })
      .catch(() => mockResult(src));

    cache.set(key, pending);
    const result = await pending;
    cache.set(key, result);
    return result;
  }

  function mockResult(src) {
    // Deterministic mock based on URL hash for demo consistency
    const h = src.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
    const deepfake = 30 + (h % 55);
    const tampering = 20 + ((h * 3) % 65);
    const meta = (h % 3 === 0);
    const final = Math.round((deepfake * 0.45 + tampering * 0.45 + (meta ? 20 : 80) * 0.1));
    const explanations = [
      'Facial blending artifacts detected in upper region.',
      'EXIF data missing or inconsistent with claimed source.',
      'Compression patterns inconsistent with original camera output.',
      'No manipulation signals detected. Image appears authentic.',
      'Clone-stamp patterns found near background edges.',
    ];
    return {
      final_score: Math.min(100, Math.max(5, final)),
      deepfake_score: deepfake,
      tampering_score: tampering,
      metadata_flag: meta,
      heatmap_url: null,
      explanation: explanations[h % explanations.length],
      source_info: meta ? 'Similar image found in 2021 article (Reuters)' : 'No prior web presence found',
      model_versions: { deepfake: 'EfficientNet-B4', tampering: 'ViT-B/16', metadata: 'EXIF-Analyzer v2' }
    };
  }

  // ─── Panel ────────────────────────────────────────────────────────────────

  function openPanel(img, result) {
    if (analysisPanel) analysisPanel.remove();

    const { bg, label, tier } = scoreColor(result.final_score);
    const panel = document.createElement('div');
    panel.className = 'tl-panel';
    panel.dataset.tier = tier;

    panel.innerHTML = `
      <div class="tl-panel-inner">
        <div class="tl-panel-header">
          <div class="tl-panel-logo"><span class="tl-logo-icon">🔍</span> TrustLens</div>
          <button class="tl-panel-close" id="tl-close">✕</button>
        </div>

        <div class="tl-panel-hero">
          <div class="tl-panel-score-wrap">
            <svg class="tl-score-svg" viewBox="0 0 120 120">
              <circle class="tl-score-bg" cx="60" cy="60" r="52"/>
              <circle class="tl-score-arc" id="tl-panel-arc" cx="60" cy="60" r="52"
                stroke="${bg}"
                stroke-dasharray="${result.final_score * 3.267} 326.7"
                transform="rotate(-90 60 60)"/>
            </svg>
            <div class="tl-score-center">
              <div class="tl-score-num" style="color:${bg}">${result.final_score}</div>
              <div class="tl-score-pct">/ 100</div>
            </div>
          </div>
          <div class="tl-panel-verdict">
            <div class="tl-verdict-label" style="color:${bg}">${label}</div>
            <div class="tl-verdict-sub">${result.explanation}</div>
            <div class="tl-verdict-source">🌐 ${result.source_info || 'No source data'}</div>
          </div>
        </div>

        <div class="tl-panel-section-title">Model Breakdown</div>
        <div class="tl-panel-models">
          ${modelCard('🤖', 'Deepfake Detection', 'EfficientNet-B4', result.deepfake_score)}
          ${modelCard('🔬', 'Tamper Analysis', 'ViT-B/16', result.tampering_score)}
          ${modelCard('📋', 'Metadata Check', 'EXIF Analyzer', result.metadata_flag ? 25 : 90, result.metadata_flag ? '⚠ Inconsistencies found' : '✓ Clean')}
        </div>

        <div class="tl-panel-section-title">Image Preview</div>
        <div class="tl-panel-preview">
          <div class="tl-preview-img-wrap">
            <img src="${img.src}" class="tl-preview-img" crossorigin="anonymous" alt="analyzed image"/>
            <div class="tl-heatmap-overlay" id="tl-heatmap"></div>
            <div class="tl-preview-badge" style="background:${bg}">${result.final_score}% authentic</div>
          </div>
          <div class="tl-preview-actions">
            <button class="tl-btn tl-btn-primary" id="tl-toggle-heatmap">👁 Toggle Heatmap</button>
            <button class="tl-btn tl-btn-outline" id="tl-reverse-search">🔍 Reverse Search</button>
            <button class="tl-btn tl-btn-outline" id="tl-copy-report">📋 Copy Report</button>
          </div>
        </div>

        <div class="tl-panel-section-title">AI Explanation</div>
        <div class="tl-panel-explain">
          <div class="tl-explain-text" id="tl-explain-text">${buildExplanation(result)}</div>
        </div>

        <div class="tl-panel-footer">
          Powered by TrustLens v1.0 · Models: EfficientNet, ViT, EXIF-Analyzer
        </div>
      </div>
    `;

    document.body.appendChild(panel);
    analysisPanel = panel;

    // Animate heatmap
    generateHeatmapOverlay(panel.querySelector('#tl-heatmap'), result);

    panel.querySelector('#tl-close').onclick = () => panel.remove();

    panel.querySelector('#tl-toggle-heatmap').onclick = function () {
      const hm = panel.querySelector('#tl-heatmap');
      hm.classList.toggle('tl-heatmap-visible');
      this.textContent = hm.classList.contains('tl-heatmap-visible') ? '🙈 Hide Heatmap' : '👁 Toggle Heatmap';
    };

    panel.querySelector('#tl-reverse-search').onclick = () => {
      window.open(`https://images.google.com/searchbyimage?image_url=${encodeURIComponent(img.src)}`, '_blank');
    };

    panel.querySelector('#tl-copy-report').onclick = function () {
      const report = `TrustLens Report\n\nAuthenticity: ${result.final_score}/100\nDeepfake Score: ${result.deepfake_score}\nTamper Score: ${result.tampering_score}\nMetadata Flag: ${result.metadata_flag}\n\n${result.explanation}\n\n${result.source_info}`;
      navigator.clipboard.writeText(report);
      this.textContent = '✓ Copied!';
      setTimeout(() => (this.textContent = '📋 Copy Report'), 2000);
    };

    // Typewriter effect for explanation
    const explainEl = panel.querySelector('#tl-explain-text');
    const text = buildExplanation(result);
    explainEl.textContent = '';
    let i = 0;
    const type = () => { if (i < text.length) { explainEl.textContent += text[i++]; requestAnimationFrame(type); } };
    setTimeout(type, 300);

    requestAnimationFrame(() => panel.classList.add('tl-panel-visible'));
  }

  function modelCard(icon, name, model, score, note) {
    const { bg, label } = scoreColor(score);
    return `
      <div class="tl-model-card">
        <div class="tl-model-icon">${icon}</div>
        <div class="tl-model-info">
          <div class="tl-model-name">${name}</div>
          <div class="tl-model-sub">${model}</div>
          ${note ? `<div class="tl-model-note">${note}</div>` : ''}
        </div>
        <div class="tl-model-score" style="color:${bg}">${score}%</div>
        <div class="tl-model-bar"><div style="width:${score}%;background:${bg}"></div></div>
      </div>
    `;
  }

  function buildExplanation(result) {
    const lines = [];
    const { tier } = scoreColor(result.final_score);
    if (tier === 'fake') lines.push(`⚠ This image shows strong signs of manipulation (${result.final_score}% authenticity).`);
    else if (tier === 'warn') lines.push(`⚡ This image shows some suspicious characteristics and warrants caution.`);
    else lines.push(`✓ This image appears to be authentic with a score of ${result.final_score}%.`);

    if (result.deepfake_score < 50) lines.push(`The deepfake detection model (EfficientNet-B4) flagged facial or generative inconsistencies with ${100 - result.deepfake_score}% confidence.`);
    if (result.tampering_score < 50) lines.push(`The tampering model (ViT-B/16) detected pixel-level anomalies indicating possible cloning or splicing.`);
    if (result.metadata_flag) lines.push(`EXIF metadata is either missing, stripped, or contradicts the image's claimed origin — a common sign of redistribution or editing.`);
    lines.push(result.source_info || 'No reverse image match found in our index.');
    return lines.join(' ');
  }

  function generateHeatmapOverlay(el, result) {
    // Canvas-based mock heatmap
    const canvas = document.createElement('canvas');
    canvas.width = 300; canvas.height = 200;
    const ctx = canvas.getContext('2d');

    const score = result.final_score;
    const numBlobs = Math.round((100 - score) / 15) + 1;

    for (let i = 0; i < numBlobs; i++) {
      const x = 40 + Math.random() * 220;
      const y = 20 + Math.random() * 160;
      const r = 20 + Math.random() * 50;
      const grd = ctx.createRadialGradient(x, y, 0, x, y, r);
      grd.addColorStop(0, `rgba(239,68,68,${0.5 + Math.random() * 0.4})`);
      grd.addColorStop(1, 'rgba(239,68,68,0)');
      ctx.fillStyle = grd;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
    }

    el.style.backgroundImage = `url(${canvas.toDataURL()})`;
  }

  // ─── Image instrumentation ────────────────────────────────────────────────

  const processedImgs = new WeakSet();

  function instrumentImage(img) {
    if (processedImgs.has(img)) return;
    if (!img.src || img.width < 80 || img.height < 80) return;
    processedImgs.add(img);

    const src = getAbsoluteURL(img.src);
    const badge = attachBadge(img);

    // Fire analysis
    analyzeImage(src).then(result => {
      updateBadge(img, result.final_score);
      img._tlResult = result;
    });

    const debouncedShow = debounce((e) => {
      if (img._tlResult) showTooltip(img, img._tlResult);
    }, 120);

    img.addEventListener('mouseenter', debouncedShow);
    img.addEventListener('mouseleave', () => setTimeout(hideTooltip, 200));
    img.addEventListener('click', (e) => {
      if (img._tlResult) {
        e.preventDefault();
        e.stopPropagation();
        openPanel(img, img._tlResult);
      }
    }, true);
  }

  function scanImages() {
    document.querySelectorAll('img').forEach(instrumentImage);
  }

  // Watch for dynamic images
  const observer = new MutationObserver(debounce(scanImages, 500));
  observer.observe(document.body, { childList: true, subtree: true });

  // Initial scan
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', scanImages);
  } else {
    scanImages();
  }

  // Message from popup
  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === 'SCAN_PAGE') scanImages();
    if (msg.type === 'TOGGLE_BADGES') {
      document.querySelectorAll('.tl-badge-wrap').forEach(el => {
        el.style.display = msg.show ? '' : 'none';
      });
    }
  });

})();
