const menubox4 = document.querySelector('.menubox4');
const menulabel4 = menubox4.querySelector('.menubox__label4');
const menuchecks4 = menubox4.querySelectorAll('input[type=checkbox]');
const menuboxRect4 = menubox4.getBoundingClientRect();
const menulabelRect4 = menulabel4.getBoundingClientRect();
const frameTime4 = 3000 / 60;
const duration4 = 350;
const frames4 = Math.ceil(duration4 / frameTime4);
const slideHeight4 = menuboxRect4.height - menulabelRect4.height;
let timer4 = null;
let currentItem4 = 0;
const debouncedFn4 = (fn) => {
  timer4 && clearTimeout(timer4);
  timer4 = setTimeout(fn, 250);
}
const documentanimateHeight44 = (e) => {
  if (e.target === menubox4 || menubox4.contains(e.target)) {
    e.stopPropagation();
    return;
  }

  debouncedFn4(() => {
    animateHeight4(true);
  });
};
document.addEventListener('click', documentanimateHeight44);
function animateHeight4 (collapsing, done) {
  let i = 0;

  function __animate4 () {
    // const scale = (collapsing ? frames4 - (i++) : i++) / frames4;
    // const height = menulabelRect4.height + (scale * slideHeight4);

    const factor = Math.pow((i++) / frames4 - 1, 3) + 1;
    const height = 2 + menulabelRect4.height + (collapsing ? 1 - factor : factor) * slideHeight4;

    menubox4.style.maxHeight = `${height}px`;

    if (i <= frames4) {
      requestAnimationFrame(__animate4);
    } else {
      if (collapsing) {
        const transitionEnded = () => {
          menubox4.removeEventListener('transitionend', transitionEnded);
          document.removeEventListener('click', documentanimateHeight44);

          (typeof done === 'function') && done();
        }

        menubox4.classList.add('menubox--collapsed');
        menubox4.addEventListener('transitionend', transitionEnded, false);
      } else {
        menuchecks4.item(currentItem4 = currentItem4 || 0).focus();
        (typeof done === 'function') && done();
      }

      timer4 && clearTimeout(timer4);
      timer4 = null;
    }
  }
  if (collapsing) {
    requestAnimationFrame(__animate4);
  } else {
    const transitionEnded = () => {
      menubox4.removeEventListener('transitionend', transitionEnded);
      requestAnimationFrame(__animate4);
    }

    menubox4.classList.remove('menubox--collapsed');
    menubox4.addEventListener('transitionend', transitionEnded, false);

    document.addEventListener('click', documentanimateHeight44);
  }
}
menulabel4.addEventListener('click', () => {
  debouncedFn4(() => {
    animateHeight4(!menubox4.classList.contains('menubox--collapsed'));
  });
});
