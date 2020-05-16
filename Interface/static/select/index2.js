const menubox2 = document.querySelector('.menubox2');
const menulabel2 = menubox2.querySelector('.menubox__label2');
const menuchecks2 = menubox2.querySelectorAll('input[type=checkbox]');
const menuboxRect2 = menubox2.getBoundingClientRect();
const menulabelRect2 = menulabel2.getBoundingClientRect();
const frameTime2 = 1000 / 60;
const duration2 = 350;
const frames2 = Math.ceil(duration2 / frameTime2);
const slideHeight2 = menuboxRect2.height - menulabelRect2.height;
let timer2 = null;
let currentItem2 = 0;
const debouncedFn2 = (fn) => {
  timer2 && clearTimeout(timer2);
  timer2 = setTimeout(fn, 250);
}
const documentanimateHeight22 = (e) => {
  if (e.target === menubox2 || menubox2.contains(e.target)) {
    e.stopPropagation();
    return;
  }
  
  debouncedFn2(() => {
    animateHeight2(true);
  });
};
document.addEventListener('click', documentanimateHeight22);
function animateHeight2 (collapsing, done) {
  let i = 0;
  
  function __animate2 () {
    // const scale = (collapsing ? frames2 - (i++) : i++) / frames2;
    // const height = menulabelRect2.height + (scale * slideHeight2);
    
    const factor = Math.pow((i++) / frames2 - 1, 3) + 1;
    const height = 2 + menulabelRect2.height + (collapsing ? 1 - factor : factor) * slideHeight2;

    menubox2.style.maxHeight = `${height}px`;

    if (i <= frames2) {
      requestAnimationFrame(__animate2);
    } else {
      if (collapsing) {
        const transitionEnded = () => {
          menubox2.removeEventListener('transitionend', transitionEnded);
          document.removeEventListener('click', documentanimateHeight22);

          (typeof done === 'function') && done();
        }

        menubox2.classList.add('menubox--collapsed');
        menubox2.addEventListener('transitionend', transitionEnded, false);
      } else {
        menuchecks2.item(currentItem2 = currentItem2 || 0).focus();
        (typeof done === 'function') && done();
      }
      
      timer2 && clearTimeout(timer2);
      timer2 = null;
    }
  }
  if (collapsing) {
    requestAnimationFrame(__animate2);
  } else {
    const transitionEnded = () => {
      menubox2.removeEventListener('transitionend', transitionEnded);
      requestAnimationFrame(__animate2);
    }
    
    menubox2.classList.remove('menubox--collapsed');
    menubox2.addEventListener('transitionend', transitionEnded, false);
      
    document.addEventListener('click', documentanimateHeight22);
  }
}
menulabel2.addEventListener('click', () => {
  debouncedFn2(() => {
    animateHeight2(!menubox2.classList.contains('menubox--collapsed'));
  });
});
