document.getElementById('analyzeBtn').addEventListener('click', () => {
    const inputText = document.getElementById('inputText').value;
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: inputText })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = `Emotion: ${data.emotion}`;
    });
});
