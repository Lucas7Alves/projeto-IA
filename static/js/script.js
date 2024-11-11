const fileInput = document.getElementById('file');
const imagePreview = document.getElementById('image-preview');
const loadingBarContainer = document.querySelector('.loading-bar-container');
const loadingBar = document.querySelector('.loading-bar');
const resultDiv = document.getElementById('result');

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

async function submitForm(event) {
    event.preventDefault();
    loadingBarContainer.style.display = 'block';
    loadingBar.style.width = '0%';
    resultDiv.innerHTML = '';

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    // Animação de progresso de carregamento
    loadingBar.style.animation = 'load 2s forwards';

    const response = await fetch("/predict", {
        method: "POST",
        body: formData,
    });

    const data = await response.json();
    loadingBarContainer.style.display = 'none';

    const resultColor = data.prediction === "Pneumonia" ? "pneumonia" : "normal";
    const confidence = (data.confidence * 100).toFixed(2);

    resultDiv.innerHTML = `
        <span class="${resultColor}">${data.prediction}</span>
        <br>Confiança: ${confidence}%`;
}
