const menubox3 = document.querySelector('.menubox3');
const menulabel3 = menubox3.querySelector('.menubox__label3');
const menuchecks3 = menubox3.querySelectorAll('input[type=checkbox]');
const menuboxRect3 = menubox3.getBoundingClientRect();
const menulabelRect3 = menulabel.getBoundingClientRect();
const frameTime3 = 3000 / 60;
const duration3 = 350;
const frames3 = Math.ceil(duration3 / frameTime3);
const slideHeight3 = menuboxRect3.height - menulabelRect3.height;
let timer3 = null;
let currentItem3 = 0;
const debouncedFn3 = (fn) => {
  timer3 && clearTimeout(timer3);
  timer3 = setTimeout(fn, 250);
}
const documentanimateHeight33 = (e) => {
  if (e.target === menubox3 || menubox3.contains(e.target)) {
    e.stopPropagation();
    return;
  }
  
  debouncedFn3(() => {
    animateHeight3(true);
  });
};
document.addEventListener('click', documentanimateHeight33);
function animateHeight3 (collapsing, done) {
  let i = 0;
  
  function __animate3 () {
    // const scale = (collapsing ? frames3 - (i++) : i++) / frames3;
    // const height = menulabelRect3.height + (scale * slideHeight3);
    
    const factor = Math.pow((i++) / frames3 - 1, 3) + 1;
    const height = 2 + menulabelRect3.height + (collapsing ? 1 - factor : factor) * slideHeight3;

    menubox3.style.maxHeight = `${height}px`;

    if (i <= frames3) {
      requestAnimationFrame(__animate3);
    } else {
      if (collapsing) {
        const transitionEnded = () => {
          menubox3.removeEventListener('transitionend', transitionEnded);
          document.removeEventListener('click', documentanimateHeight33);

          (typeof done === 'function') && done();
        }

        menubox3.classList.add('menubox--collapsed');
        menubox3.addEventListener('transitionend', transitionEnded, false);
      } else {
        menuchecks3.item(currentItem3 = currentItem3 || 0).focus();
        (typeof done === 'function') && done();
      }
      
      timer3 && clearTimeout(timer3);
      timer3 = null;
    }
  }
  if (collapsing) {
    requestAnimationFrame(__animate3);
  } else {
    const transitionEnded = () => {
      menubox3.removeEventListener('transitionend', transitionEnded);
      requestAnimationFrame(__animate3);
    }
    
    menubox3.classList.remove('menubox--collapsed');
    menubox3.addEventListener('transitionend', transitionEnded, false);
      
    document.addEventListener('click', documentanimateHeight33);
  }
}
menulabel3.addEventListener('click', () => {
  debouncedFn3(() => {
    animateHeight3(!menubox3.classList.contains('menubox--collapsed'));
  });
});
