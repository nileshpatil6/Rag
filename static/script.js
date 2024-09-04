function uploadPDF() {
    const fileInput = document.getElementById('pdfFile');
    const file = fileInput.files[0];
    if (file) {
        const formData = new FormData();
        formData.append('file', file);
        
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            alert(data.message);
        })
        .catch(error => {
            console.error('Error:', error);
        });
    } else {
        alert('Please select a PDF file');
    }
}

function askQuestion() {
    const questionInput = document.getElementById('question');
    const question = questionInput.value.trim();
    if (question) {
        fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question }),
        })
        .then(response => response.json())
        .then(data => {
            const chatDiv = document.getElementById('chat');
            chatDiv.innerHTML += `<p><strong>You:</strong> ${question}</p>`;
            chatDiv.innerHTML += `<p><strong>Bot:</strong> ${data.answer}</p>`;
            chatDiv.scrollTop = chatDiv.scrollHeight;
            questionInput.value = '';
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
}
