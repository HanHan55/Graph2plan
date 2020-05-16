const menubox1 = document.querySelector('.menubox1');
const menulabel1 = menubox1.querySelector('.menubox__label1');
const menuchecks1 = menubox1.querySelectorAll('input[type=checkbox]');
const menuboxRect1 = menubox1.getBoundingClientRect();
const menulabelRect1 = menulabel.getBoundingClientRect();
const frameTime1 = 1000 / 60;
const duration1 = 350;
const frames1 = Math.ceil(duration1 / frameTime1);
const slideHeight1 = menuboxRect1.height - menulabelRect1.height;
let timer1 = null;
let currentItem1 = 0;
const debouncedFn1 = (fn) => {
  timer1 && clearTimeout(timer1);
  timer1 = setTimeout(fn, 250);
}
const documentanimateHeight11 = (e) => {
  if (e.target === menubox1 || menubox1.contains(e.target)) {
    e.stopPropagation();
    return;
  }
  
  debouncedFn1(() => {
    animateHeight1(true);
  });
};
document.addEventListener('click', documentanimateHeight11);
function animateHeight1 (collapsing, done) {
  let i = 0;
  
  function __animate1 () {
    // const scale = (collapsing ? frames1 - (i++) : i++) / frames1;
    // const height = menulabelRect1.height + (scale * slideHeight1);
    
    const factor = Math.pow((i++) / frames1 - 1, 3) + 1;
    const height = 2 + menulabelRect1.height + (collapsing ? 1 - factor : factor) * slideHeight1;

    menubox1.style.maxHeight = `${height}px`;

    if (i <= frames1) {
      requestAnimationFrame(__animate1);
    } else {
      if (collapsing) {
        const transitionEnded = () => {
          menubox1.removeEventListener('transitionend', transitionEnded);
          document.removeEventListener('click', documentanimateHeight11);

          (typeof done === 'function') && done();
        }

        menubox1.classList.add('menubox--collapsed');
        menubox1.addEventListener('transitionend', transitionEnded, false);
      } else {
        menuchecks1.item(currentItem1 = currentItem1 || 0).focus();
        (typeof done === 'function') && done();
      }
      
      timer1 && clearTimeout(timer1);
      timer1 = null;
    }
  }
  if (collapsing) {
    requestAnimationFrame(__animate1);
  } else {
    const transitionEnded = () => {
      menubox1.removeEventListener('transitionend', transitionEnded);
      requestAnimationFrame(__animate1);
    }
    
    menubox1.classList.remove('menubox--collapsed');
    menubox1.addEventListener('transitionend', transitionEnded, false);
      
    document.addEventListener('click', documentanimateHeight11);
  }
}
menulabel1.addEventListener('click', () => {
  debouncedFn1(() => {
    animateHeight1(!menubox1.classList.contains('menubox--collapsed'));
  });
});
