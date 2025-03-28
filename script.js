// DOM Elements
const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const originalCanvas = document.getElementById('originalCanvas');
const processedCanvas = document.getElementById('processedCanvas');
const edgeThreshold = document.getElementById('edgeThreshold');
const simplifyToggle = document.getElementById('simplifyToggle');
const processBtn = document.getElementById('processBtn');
const downloadPng = document.getElementById('downloadPng');
const downloadSvg = document.getElementById('downloadSvg');

// Canvas contexts
const originalCtx = originalCanvas.getContext('2d');
const processedCtx = processedCanvas.getContext('2d');

// Global variables
let originalImage = null;
let processedImageData = null;

// Set consistent styling for processed output
function setupProcessedCanvasStyle() {
    processedCtx.fillStyle = '#ffffff';
    processedCtx.fillRect(0, 0, processedCanvas.width, processedCanvas.height);
    processedCtx.strokeStyle = '#000000';
    processedCtx.lineWidth = 1.5;
}

// Initialize image processing tools
async function initializeImageTools() {
    const loadingIndicator = document.getElementById('loadingIndicator');
    loadingIndicator.style.display = 'block';
    
    try {
        if (!window.cv) {
            await loadOpenCV();
            console.log('Using OpenCV for advanced processing');
        }
    } catch (error) {
        console.log('Using basic Canvas processing');
    } finally {
        loadingIndicator.style.display = 'none';
    }
}

// Handle file selection
function handleFileSelect(file) {
    if (!file.type.match('image.*')) {
        alert('Please select an image file');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            originalImage = img;
            drawOriginalImage();
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// Draw original image on canvas
function drawOriginalImage() {
    const maxWidth = 500;
    const maxHeight = 500;
    let width = originalImage.width;
    let height = originalImage.height;

    if (width > height) {
        if (width > maxWidth) {
            height *= maxWidth / width;
            width = maxWidth;
        }
    } else {
        if (height > maxHeight) {
            width *= maxHeight / height;
            height = maxHeight;
        }
    }

    originalCanvas.width = width;
    originalCanvas.height = height;
    processedCanvas.width = width;
    processedCanvas.height = height;

    originalCtx.drawImage(originalImage, 0, 0, width, height);
}

// Process image with edge detection
async function processImage() {
    if (!originalImage) {
        alert('Please upload an image first');
        return;
    }

    try {
        processBtn.disabled = true;
        processBtn.textContent = 'Processing...';
        console.log('Starting image processing...');
        
        await initializeImageTools();

        if (window.cv) {
            // OpenCV processing
            const src = cv.imread(originalCanvas);
            const gray = new cv.Mat();
            const edges = new cv.Mat();
            
            cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
            const thresholdValue = parseInt(edgeThreshold.value);
            cv.Canny(gray, edges, thresholdValue * 0.5, thresholdValue);
            
            if (simplifyToggle.checked) {
                setupProcessedCanvasStyle();
                // Find contours
                const contours = new cv.MatVector();
                const hierarchy = new cv.Mat();
                cv.findContours(edges.clone(), contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

                for (let i = 0; i < contours.size(); ++i) {
                    const contour = contours.get(i);
                    const epsilon = 0.01 * cv.arcLength(contour, true);
                    const approx = new cv.Mat();
                    cv.approxPolyDP(contour, approx, epsilon, true);

                    processedCtx.beginPath();
                    for (let j = 0; j < approx.rows; j++) {
                        const point = approx.data32S.subarray(j * 2, j * 2 + 2);
                        if (j === 0) {
                            processedCtx.moveTo(point[0], point[1]);
                        } else {
                            processedCtx.lineTo(point[0], point[1]);
                        }
                    }
                    processedCtx.closePath();
                    processedCtx.stroke();
                    approx.delete();
                }
                contours.delete();
                hierarchy.delete();
            } else {
                setupProcessedCanvasStyle();
                cv.imshow(processedCanvas, edges);
            }
            
            src.delete();
            gray.delete();
            edges.delete();
        } else {
            // Canvas fallback processing
            const thresholdValue = parseInt(edgeThreshold.value) / 255;
            const imageData = originalCtx.getImageData(0, 0, originalCanvas.width, originalCanvas.height);
            processedCtx.clearRect(0, 0, processedCanvas.width, processedCanvas.height);
            
            // Improved edge detection using Canvas
            processedCtx.strokeStyle = '#000000';
            processedCtx.lineWidth = 1.5;
            processedCtx.fillStyle = '#ffffff';
            processedCtx.fillRect(0, 0, processedCanvas.width, processedCanvas.height);
            
            // Improved edge detection with Sobel operator approximation
            const pixels = imageData.data;
            const width = originalCanvas.width;
            const height = originalCanvas.height;
            
            // Create white background
            processedCtx.fillStyle = '#ffffff';
            processedCtx.fillRect(0, 0, width, height);
            processedCtx.strokeStyle = '#000000';
            processedCtx.lineWidth = 1.5;
            
            // Edge detection
            for (let y = 1; y < height - 1; y++) {
                for (let x = 1; x < width - 1; x++) {
                    const i = (y * width + x) * 4;
                    
                    // Get surrounding pixels
                    const top = (y-1)*width*4 + x*4;
                    const bottom = (y+1)*width*4 + x*4;
                    const left = y*width*4 + (x-1)*4;
                    const right = y*width*4 + (x+1)*4;
                    
                    // Calculate gradient
                    const gx = -pixels[top] - 2*pixels[left] - pixels[bottom] + 
                               pixels[top+4] + 2*pixels[right] + pixels[bottom+4];
                    const gy = -pixels[top] - 2*pixels[top+4] - pixels[top+8] + 
                               pixels[bottom] + 2*pixels[bottom+4] + pixels[bottom+8];
                    
                    const magnitude = Math.sqrt(gx*gx + gy*gy);
                    if (magnitude > thresholdValue * 1000) {
                        processedCtx.beginPath();
                        processedCtx.moveTo(x, y);
                        processedCtx.lineTo(x+1, y+1);
                        processedCtx.stroke();
                    }
                }
            }
        }
        
        console.log('Image processing completed');
    } catch (error) {
        console.error('Processing error:', error);
        alert('Error processing image: ' + error.message);
    } finally {
        processBtn.disabled = false;
        processBtn.textContent = 'Process Image';
    }
}

// Download processed image as PNG
function downloadAsPng() {
    if (!processedCanvas.toDataURL().includes('image/png')) {
        alert('Please process an image first');
        return;
    }

    const link = document.createElement('a');
    link.download = 'traced-image.png';
    link.href = processedCanvas.toDataURL('image/png');
    link.click();
}

// Download processed image as SVG
function downloadAsSvg() {
    if (!simplifyToggle.checked) {
        alert('SVG download only available with simplified shapes');
        return;
    }

    // This would be implemented with more complex SVG generation
    alert('SVG export will be implemented in the next version');
}

// Event listeners
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFileSelect(e.target.files[0]);
    }
});

dropZone.addEventListener('click', () => {
    fileInput.click();
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#4a6fa5';
    dropZone.style.backgroundColor = 'rgba(74, 111, 165, 0.1)';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = 'var(--border-color)';
    dropZone.style.backgroundColor = '';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--border-color)';
    dropZone.style.backgroundColor = '';

    if (e.dataTransfer.files.length) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

processBtn.addEventListener('click', processImage);
downloadPng.addEventListener('click', downloadAsPng);
downloadSvg.addEventListener('click', downloadAsSvg);

// Theme toggle
const themeToggle = document.createElement('button');
themeToggle.className = 'theme-toggle';
themeToggle.innerHTML = '🌓';
themeToggle.addEventListener('click', () => {
    document.body.dataset.theme = document.body.dataset.theme === 'dark' ? 'light' : 'dark';
    localStorage.setItem('theme', document.body.dataset.theme);
});
document.body.appendChild(themeToggle);

// Initialize theme
const savedTheme = localStorage.getItem('theme') || 'light';
document.body.dataset.theme = savedTheme;