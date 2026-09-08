// TrustLens Background Service Worker

const API_BASE = 'http://localhost:8000';

chrome.runtime.onInstalled.addListener(() => {
  console.log('[TrustLens] Installed & ready.');
});

// Context menu for right-click on images
chrome.contextMenus?.create({
  id: 'trustlens-analyze',
  title: '🔍 Analyze with TrustLens',
  contexts: ['image'],
});

chrome.contextMenus?.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'trustlens-analyze' && info.srcUrl) {
    chrome.tabs.sendMessage(tab.id, {
      type: 'ANALYZE_URL',
      url: info.srcUrl,
    });
  }
});

// Relay messages between popup and content script
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === 'GET_PAGE_STATS') {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      chrome.tabs.sendMessage(tabs[0].id, { type: 'PAGE_STATS_REQUEST' }, (resp) => {
        sendResponse(resp || { total: 0, fake: 0, warn: 0, real: 0 });
      });
    });
    return true; // async
  }
});
