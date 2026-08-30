// Scroll-spy: highlight the nav link for whichever section is in view
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-link');

const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      const id = entry.target.getAttribute('id');
      navLinks.forEach((link) => {
        link.classList.toggle('active', link.dataset.section === id);
      });
    }
  });
}, { rootMargin: '-40% 0px -55% 0px' });

sections.forEach((section) => observer.observe(section));

// Hero panel: animate checks ticking on one by one, replayed when scrolled into view
const checkRows = document.querySelectorAll('#check-list .check-row');
let hasPlayed = false;

function playCheckAnimation() {
  if (hasPlayed) return;
  hasPlayed = true;
  checkRows.forEach((row, i) => {
    setTimeout(() => {
      row.classList.add('done');
      row.querySelector('.check-icon').textContent = '✓';
    }, i * 420);
  });
}

const heroObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) playCheckAnimation();
  });
}, { threshold: 0.4 });

const heroPanel = document.querySelector('.hero-panel');
if (heroPanel) heroObserver.observe(heroPanel);

// Chat popover toggle
const chatFab = document.getElementById('chat-fab');
const chatPopover = document.getElementById('chat-popover');
if (chatFab && chatPopover) {
  chatFab.addEventListener('click', () => {
    chatPopover.classList.toggle('open');
  });
}