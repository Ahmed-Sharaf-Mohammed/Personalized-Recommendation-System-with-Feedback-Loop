/**
 * tracker.js — Frontend event tracking helpers
 * Logs browsing events (view/click/cart) to Django AJAX API.
 * Schema-compatible with browsing_logs for ML retraining.
 */

// ── CSRF ─────────────────────────────────────────────────────────────────────
function getCsrf() {
  const c = document.cookie.split(';').find(x => x.trim().startsWith('csrftoken='));
  return c ? c.trim().split('=')[1] : '';
}

// ── Core fetch ────────────────────────────────────────────────────────────────
async function trackEvent(itemId, eventType = 'view', source = 'direct') {
  try {
    const res = await fetch('/api/track/', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCsrf() },
      body:    JSON.stringify({ item_id: itemId, event_type: eventType, source }),
    });
    if (res.ok) {
      const data = await res.json();
      if (data.cart_count !== undefined) updateCartBadge(data.cart_count);
    }
  } catch (e) { /* silent */ }
}

// ── Cart helpers ──────────────────────────────────────────────────────────────
function trackAndCart(btn) {
  const itemId = btn.dataset.itemId;
  const source = btn.closest('[data-source]')?.dataset.source || 'direct';
  if (!itemId) return;
  trackEvent(itemId, 'add_to_cart', source);
  flashCartBtn(btn);
}

function trackAndCartById(itemId, source = 'direct') {
  const btn = document.getElementById('add-cart-btn');
  trackEvent(itemId, 'add_to_cart', source);
  if (btn) flashCartBtn(btn);
}

function flashCartBtn(btn) {
  const orig = btn.innerHTML;
  btn.innerHTML = '<i class="bi bi-check-lg me-1"></i>Added!';
  btn.classList.replace('btn-warning', 'btn-success');
  setTimeout(() => {
    btn.innerHTML = orig;
    btn.classList.replace('btn-success', 'btn-warning');
  }, 1400);
}

// ── Badge ─────────────────────────────────────────────────────────────────────
function updateCartBadge(count) {
  const badge = document.getElementById('cart-badge');
  if (!badge) return;
  badge.textContent = count;
  badge.style.display = count > 0 ? '' : 'none';
}

// ── Auto-track product card clicks ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.product-card-link').forEach(link => {
    link.addEventListener('click', () => {
      const card   = link.closest('[data-item-id]');
      const itemId = card?.dataset.itemId;
      const source = card?.dataset.source || 'direct';
      if (itemId) trackEvent(itemId, 'click', source);
    });
  });
});
