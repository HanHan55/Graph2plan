const menubox = document.querySelector('.menubox');
const menulabel = menubox.querySelector('.menubox__label');
const menuchecks = menubox.querySelectorAll('input[type=checkbox]');
const menuboxRect = menubox.getBoundingClientRect();
const menulabelRect = menulabel.getBoundingClientRect();
const frameTime = 1000 / 60;
const duration = 350;
const frames = Math.ceil(duration / frameTime);
const slideHeight = menuboxRect.height - menulabelRect.height;
let timer = null;
let currentItem = 0;
const debouncedFn = (fn) => {
  timer && clearTimeout(timer);
  timer = setTimeout(fn, 250);
}
const documentAnimateHeight = (e) => {
  if (e.target === menubox || menubox.contains(e.target)) {
    e.stopPropagation();
    return;
  }
  
  debouncedFn(() => {
    animateHeight(true);
  });
};
document.addEventListener('click', documentAnimateHeight);
function animateHeight (collapsing, done) {
  let i = 0;
  
  function __animate () {
    // const scale = (collapsing ? frames - (i++) : i++) / frames;
    // const height = menulabelRect.height + (scale * slideHeight);
    
    const factor = Math.pow((i++) / frames - 1, 3) + 1;
    const height = 2 + menulabelRect.height + (collapsing ? 1 - factor : factor) * slideHeight;

    menubox.style.maxHeight = `${height}px`;

    if (i <= frames) {
      requestAnimationFrame(__animate);
    } else {
      if (collapsing) {
        const transitionEnded = () => {
          menubox.removeEventListener('transitionend', transitionEnded);
          document.removeEventListener('click', documentAnimateHeight);

          (typeof done === 'function') && done();
        }

        menubox.classList.add('menubox--collapsed');
        menubox.addEventListener('transitionend', transitionEnded, false);
      } else {
        menuchecks.item(currentItem = currentItem || 0).focus();
        (typeof done === 'function') && done();
      }
      
      timer && clearTimeout(timer);
      timer = null;
    }
  }
  if (collapsing) {
    requestAnimationFrame(__animate);
  } else {
    const transitionEnded = () => {
      menubox.removeEventListener('transitionend', transitionEnded);
      requestAnimationFrame(__animate);
    }
    
    menubox.classList.remove('menubox--collapsed');
    menubox.addEventListener('transitionend', transitionEnded, false);
      
    document.addEventListener('click', documentAnimateHeight);
  }
}
menulabel.addEventListener('click', () => {
  debouncedFn(() => {
    animateHeight(!menubox.classList.contains('menubox--collapsed'));
  });
});
