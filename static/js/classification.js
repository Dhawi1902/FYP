function classifyImage() {
    const resultText = document.getElementById("resultText");
    const validationOptions = document.getElementById("validationOptions");
    const correctClassInput = document.getElementById("correctClass");

    // Simulate a classification result
    const predictedClass = "Caries"; // Replace with actual model prediction
    resultText.innerText = `Predicted Class: ${predictedClass}`;
    validationOptions.style.display = "block";
    correctClassInput.style.display = "none"; // Hide the correct class input initially
}

function acceptPrediction() {
    const resultText = document.getElementById("resultText").innerText;
    addToHistory(resultText);
    resetClassification();
}

function rejectPrediction() {
    const correctClassInput = document.getElementById("correctClass");
    correctClassInput.style.display = "block"; // Show the input for the correct class
    correctClassInput.focus();
}

function submitCorrectClass() {
    const correctClass = document.getElementById("correctClass").value;
    if (correctClass) {
        const resultText = `Corrected Class: ${correctClass}`;
        addToHistory(resultText);
        resetClassification();
    }
}

function addToHistory(text) {
    const historyList = document.getElementById("historyList");
    const newItem = document.createElement("li");
    newItem.innerText = text;
    historyList.appendChild(newItem);
}

function resetClassification() {
    document.getElementById("resultText").innerText = "No result yet.";
    document.getElementById("validationOptions").style.display = "none";
    document.getElementById("correctClass").value = "";
    document.getElementById("correctClass").style.display = "none";
}


// Trigger file input when drop zone is clicked
document.getElementById("drop-zone").addEventListener("click", function() {
    document.getElementById("fileInput").click();
});

function previewImage(event) {
    const file = event.target.files[0];
    const preview = document.getElementById("imagePreview");

    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = "block";  // Show the image
        };
        reader.readAsDataURL(file);
    } else {
        preview.style.display = "none";  // Hide the preview if no file is selected
    }
}


// Trigger file input when drop zone is clicked
document.getElementById("drop-zone").addEventListener("click", function() {
    document.getElementById("fileInput").click();
});

// Handle drag-and-drop functionality
const dropZone = document.getElementById("drop-zone");

dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();  // Prevent default to allow drop
    dropZone.classList.add("drag-over");  // Add styling on drag
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");  // Remove styling when not dragging over
});

dropZone.addEventListener("drop", (event) => {
    event.preventDefault();  // Prevent browser’s default behavior for dropped files
    dropZone.classList.remove("drag-over");

    const files = event.dataTransfer.files;
    if (files.length > 0) {
        previewImage({ target: { files: files } });  // Preview the dropped file
    }
});

// Preview image function
function previewImage(event) {
    const file = event.target.files[0];
    const preview = document.getElementById("imagePreview");

    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = "block";  // Show the image
        };
        reader.readAsDataURL(file);
    } else {
        preview.style.display = "none";  // Hide the preview if no file is selected
    }
}

// Function to preview the uploaded image
function previewImage(event) {
    const file = event.target.files[0];
    const previewContainer = document.getElementById("imagePreview");

    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            previewContainer.style.backgroundImage = `url(${e.target.result})`;
            previewContainer.style.backgroundSize = "cover";
        };
        reader.readAsDataURL(file);
    }
}

// Function to classify the image
function classifyImage() {
    const fileInput = document.getElementById("imageUpload");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please upload an image first.");
        return;
    }

    document.getElementById("prediction-result").textContent = "Loading...";
    document.getElementById("validation-options").style.display = "none";

    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
        method: "POST",
        body: formData,
    })
    .then(response => response.json())
    .then(result => {
        document.getElementById("prediction-result").textContent = `Prediction: ${result.prediction}`;
        document.getElementById("validation-options").style.display = "block";
    })
    .catch(error => {
        console.error("Error:", error);
        document.getElementById("prediction-result").textContent = "Error in classification.";
    });
}

// Function to validate the prediction
function validatePrediction(isCorrect) {
    if (isCorrect) {
        alert("Thank you for confirming!");
    } else {
        document.getElementById("correction-input").style.display = "block";
        document.getElementById("correction-input").placeholder = "Enter correct class";
    }
}
