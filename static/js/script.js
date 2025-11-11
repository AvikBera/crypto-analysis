// Wait for the DOM to be fully loaded before running scripts
document.addEventListener("DOMContentLoaded", function() {
    
    // Find the file input and the filename display element
    const fileInput = document.getElementById("file-upload-input");
    const filenameDisplay = document.getElementById("file-upload-filename");
    const submitButton = document.getElementById("submit-button");

    if (fileInput && filenameDisplay) {
        fileInput.addEventListener("change", function() {
            // Check if files were selected
            if (fileInput.files.length > 0) {
                if (fileInput.files.length === 1) {
                    // Show single filename
                    filenameDisplay.textContent = fileInput.files[0].name;
                } else {
                    // Show number of files
                    filenameDisplay.textContent = `${fileInput.files.length} files selected`;
                }
                
                // Enable the submit button if it exists
                if (submitButton) {
                    submitButton.disabled = false;
                }
            } else {
                // No files selected
                filenameDisplay.textContent = "No files chosen...";
                
                // Disable the submit button if it exists
                if (submitButton) {
                    submitButton.disabled = true;
                }
            }
        });
    }

});