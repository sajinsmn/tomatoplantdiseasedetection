// Handle image upload
document.getElementById('uploadButton').addEventListener('click', function() {
    document.getElementById('imageUpload').click();
});

document.getElementById('imageUpload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('uploadedImage').src = e.target.result;
            document.getElementById('uploadedImageContainer').style.display = 'block';
        }
        reader.readAsDataURL(file);
    }
});

// Handle photo capture
document.getElementById('captureButton').addEventListener('click', function() {
    document.getElementById('cameraInput').click();
});

document.getElementById('cameraInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('uploadedImage').src = e.target.result;
            document.getElementById('uploadedImageContainer').style.display = 'block';
        }
        reader.readAsDataURL(file);
    }
});

// Handle cancel upload
document.getElementById('cancelButton').addEventListener('click', function() {
    document.getElementById('uploadedImage').src = '';
    document.getElementById('uploadedImageContainer').style.display = 'none';
    document.getElementById('imageUpload').value = ''; // Clear the input
    document.getElementById('cameraInput').value = ''; // Clear the camera input
});

// Handle check disease button click
document.getElementById('checkDiseaseButton').addEventListener('click', function() {
    const imageSrc = document.getElementById('uploadedImage').src;
    if (imageSrc) {
        // Here you would typically call your disease detection model
        // For demonstration, we'll use a simple random check
        const isDiseased = Math.random() < 0.5; // Randomly determine if diseased or healthy
        const resultText = isDiseased ? "Diseased" : "Healthy";
        document.getElementById('diseaseResult').innerText = resultText;
    } else {
        document.getElementById('diseaseResult').innerText = "Please upload an image first.";
    }
});