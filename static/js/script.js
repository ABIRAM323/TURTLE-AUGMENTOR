let datasetFile = null;

function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a file to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        // Display preview
        const previewDiv = document.getElementById('preview');
        previewDiv.innerHTML = '';
        data.preview.forEach(item => {
            const qaDiv = document.createElement('div');
            qaDiv.innerHTML = `<strong>Q:</strong> ${item.question}<br><strong>A:</strong> ${item.answer}<br><br>`;
            previewDiv.appendChild(qaDiv);
        });

        // Enable download button
        datasetFile = data.dataset_file;
        document.getElementById('downloadBtn').disabled = false;
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while processing the file.');
    });
}

function downloadFile() {
    if (!datasetFile) {
        alert('No dataset available to download.');
        return;
    }

    window.location.href = `/download/${datasetFile}`;
}
