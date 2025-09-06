document.getElementById('predictionForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const resultDiv = document.getElementById('result');
    const resultContent = document.getElementById('resultContent');
    
    // Show loading
    resultContent.innerHTML = '<p>Making prediction...</p>';
    resultDiv.style.display = 'block';
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            const riskClass = data.risk_level.toLowerCase().replace(' ', '-');
            resultContent.innerHTML = `
                <div class="probability ${riskClass}">${data.probability}%</div>
                <h3>Risk Level: <span class="${riskClass}">${data.risk_level}</span></h3>
                <p><strong>Diagnosis:</strong> ${data.prediction === 1 ? 'Heart Disease Risk Detected' : 'Low Heart Disease Risk'}</p>
                <div style="margin-top: 20px; padding: 15px; background: #e9ecef; border-radius: 5px;">
                    <small><strong>Disclaimer:</strong> This prediction is for educational purposes only. 
                    Please consult with healthcare professionals for actual medical diagnosis.</small>
                </div>
            `;
        } else {
            resultContent.innerHTML = `<p style="color: red;">Error: ${data.message}</p>`;
        }
    })
    .catch(error => {
        resultContent.innerHTML = '<p style="color: red;">Error making prediction</p>';
        console.error('Error:', error);
    });
});