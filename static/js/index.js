// Seleciona os elementos que devem ser animados
const animatedElements = document.querySelectorAll('[data-animate]');

// Função para verificar se o elemento está visível na tela
function handleScroll() {
    const triggerHeight = window.innerHeight * 0.8; // 80% da altura da janela
    animatedElements.forEach((element) => {
        const elementTop = element.getBoundingClientRect().top;
        if (elementTop < triggerHeight) {
            element.classList.add('visible');
        }
    });
}

// Escuta o evento de rolagem
window.addEventListener('scroll', handleScroll);

// Inicializa a animação ao carregar a página
handleScroll();
