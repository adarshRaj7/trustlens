// TrustLens Popup Script
const API_BASE = 'http://localhost:8000';

async function checkAPI() {
  try {
    const r = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2500) });
    const data = await r.json();
    document.getElementById('api-dot').className = 'api-dot ok';
    document.getElementById('api-label').textContent = `Connected · ${data.models_loaded || 3} models loaded`;
  } catch {
    document.getElementById('api-dot').className = 'api-dot err';
    document.getElementById('api-label').textContent = 'Offline — using mock mode';
  }
}

document.getElementById('scan-btn').onclick = () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.tabs.sendMessage(tabs[0].id, { type: 'SCAN_PAGE' });
  });
  window.close();
};

document.getElementById('toggle-badges').onchange = (e) => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.tabs.sendMessage(tabs[0].id, { type: 'TOGGLE_BADGES', show: e.target.checked });
  });
};

// Load stats from storage
chrome.storage.local.get(['stats'], ({ stats }) => {
  if (stats) {
    document.getElementById('stat-real').textContent = stats.real ?? '–';
    document.getElementById('stat-warn').textContent = stats.warn ?? '–';
    document.getElementById('stat-fake').textContent = stats.fake ?? '–';
  }
});

checkAPI();
