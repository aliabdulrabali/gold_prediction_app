// Gold Price Prediction Dashboard - Main JavaScript
document.addEventListener('DOMContentLoaded', initApp);

// Global variables
const API_BASE_URL = '';  // Empty for same origin
let historyChartInstance = null;
let predictionChartInstance = null;
let currentPage = 1;
let pageSize = 10;
let totalRecords = 0;
let historyData = [];
let toast;

// Initialize the application
async function initApp() {
    console.log('Initializing Gold Price Prediction Dashboard');
    
    // Initialize Bootstrap components
    initializeBootstrapComponents();
    
    // Set up event listeners
    setupEventListeners();
    
    // Initialize date filters with default values
    initializeDateFilters();
    
    // Load initial data
    await Promise.all([
        fetchCurrentPrice(),
        fetchHistoricalData(),
        fetchModelsInfo()
    ]);
    
    // Fetch predictions with default settings
    await fetchPredictions();
    
    // Show a welcome toast
    showToast('Dashboard Initialized', 'Gold price data and predictions have been loaded successfully.');
}

// Initialize Bootstrap components
function initializeBootstrapComponents() {
    // Initialize toast
    toast = new bootstrap.Toast(document.getElementById('toast'));
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Set up event listeners
function setupEventListeners() {
    // Model selection change
    document.getElementById('model-select').addEventListener('change', fetchPredictions);
    
    // Days selection change
    document.getElementById('days-select').addEventListener('change', fetchPredictions);
    
    // Time range buttons
    document.querySelectorAll('.time-range-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            // Update active state
            document.querySelectorAll('.time-range-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            
            // Fetch data for selected range
            const days = parseInt(e.target.dataset.range);
            fetchHistoricalData(days);
        });
    });
    
    // Refresh button
    document.getElementById('refresh-btn').addEventListener('click', refreshAllData);
    
    // Train models button
    document.getElementById('train-btn').addEventListener('click', trainModels);
    
    // Pagination buttons
    document.getElementById('prev-page').addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            updateHistoryTable();
        }
    });
    
    document.getElementById('next-page').addEventListener('click', () => {
        if (currentPage * pageSize < totalRecords) {
            currentPage++;
            updateHistoryTable();
        }
    });
    
    // Filter button
    document.getElementById('filter-btn').addEventListener('click', () => {
        const startDate = document.getElementById('start-date').value;
        const endDate = document.getElementById('end-date').value;
        fetchHistoricalData(null, startDate, endDate);
    });
    
    // Navigation links smooth scrolling
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Initialize date filters with default values
function initializeDateFilters() {
    const today = new Date();
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(today.getDate() - 30);
    
    const endDateInput = document.getElementById('end-date');
    const startDateInput = document.getElementById('start-date');
    
    endDateInput.value = formatDateForInput(today);
    startDateInput.value = formatDateForInput(thirtyDaysAgo);
    
    // Set max date to today
    endDateInput.max = formatDateForInput(today);
    startDateInput.max = formatDateForInput(today);
}

// Format date for input fields (YYYY-MM-DD)
function formatDateForInput(date) {
    return date.toISOString().split('T')[0];
}

// Fetch current gold price
async function fetchCurrentPrice() {
    try {
        const priceDisplay = document.getElementById('price-display');
        const priceChange = document.getElementById('price-change');
        const priceTimestamp = document.getElementById('price-timestamp');
        const priceHigh = document.getElementById('price-high');
        const priceLow = document.getElementById('price-low');
        const priceOpen = document.getElementById('price-open');
        const priceClose = document.getElementById('price-close');
        
        // Show loading state
        priceDisplay.innerHTML = '<div class="spinner-border text-warning" role="status"><span class="visually-hidden">Loading...</span></div>';
        
        const response = await fetch(`${API_BASE_URL}/current`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            const data = result.data;
            
            // Update price display
            priceDisplay.textContent = `$${data.price.toFixed(2)} ${data.currency || 'USD'}`;
            
            // Update price change
            if (data.ch !== null && data.ch !== undefined) {
                const changeValue = data.ch.toFixed(2);
                const changePercent = data.chp ? data.chp.toFixed(2) : '0.00';
                const isPositive = data.ch >= 0;
                
                priceChange.textContent = `${isPositive ? '+' : ''}${changeValue} (${isPositive ? '+' : ''}${changePercent}%)`;
                priceChange.className = `price-change ${isPositive ? 'positive' : 'negative'}`;
                priceChange.innerHTML = `${isPositive ? '<i class="bi bi-arrow-up-right"></i>' : '<i class="bi bi-arrow-down-right"></i>'} ${priceChange.textContent}`;
            } else {
                priceChange.textContent = 'No change data available';
                priceChange.className = 'price-change';
            }
            
            // Update timestamp
            const timestamp = new Date(data.timestamp * 1000);
            priceTimestamp.textContent = `Last updated: ${timestamp.toLocaleString()}`;
            
            // Update price statistics
            priceHigh.textContent = data.high ? `$${data.high.toFixed(2)}` : '--';
            priceLow.textContent = data.low ? `$${data.low.toFixed(2)}` : '--';
            priceOpen.textContent = data.open ? `$${data.open.toFixed(2)}` : '--';
            priceClose.textContent = data.close ? `$${data.close.toFixed(2)}` : '--';
        } else {
            throw new Error('Failed to fetch current price');
        }
    } catch (error) {
        console.error('Error fetching current price:', error);
        document.getElementById('price-display').textContent = 'Error loading price';
        showToast('Error', 'Failed to fetch current gold price. Please try again later.', 'error');
    }
}

// Fetch historical gold price data
async function fetchHistoricalData(days = 30, startDate = null, endDate = null) {
    try {
        // Construct query parameters
        let url = `${API_BASE_URL}/history?limit=1000`;
        
        if (startDate && endDate) {
            url += `&start_date=${startDate}&end_date=${endDate}`;
        }
        
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            // Store the full dataset
            historyData = result.data;
            totalRecords = historyData.length;
            
            // Filter by days if specified and no date range was provided
            if (days && (!startDate || !endDate)) {
                const cutoffDate = new Date();
                cutoffDate.setDate(cutoffDate.getDate() - days);
                historyData = historyData.filter(item => new Date(item.date) >= cutoffDate);
            }
            
            // Sort by date (oldest to newest)
            historyData.sort((a, b) => new Date(a.date) - new Date(b.date));
            
            // Update the history chart
            updateHistoryChart(historyData);
            
            // Update the history table
            updateHistoryTable();
        } else {
            throw new Error('Failed to fetch historical data');
        }
    } catch (error) {
        console.error('Error fetching historical data:', error);
        showToast('Error', 'Failed to fetch historical gold price data. Please try again later.', 'error');
    }
}

// Update the history chart with data
function updateHistoryChart(data) {
    const ctx = document.getElementById('historyChart').getContext('2d');
    
    // Prepare data for chart
    const labels = data.map(item => item.date);
    const prices = data.map(item => item.price);
    
    // Calculate moving average (7-day)
    const movingAvgPeriod = 7;
    const movingAvg = [];
    
    for (let i = 0; i < prices.length; i++) {
        if (i < movingAvgPeriod - 1) {
            movingAvg.push(null);
        } else {
            const sum = prices.slice(i - movingAvgPeriod + 1, i + 1).reduce((a, b) => a + b, 0);
            movingAvg.push(sum / movingAvgPeriod);
        }
    }
    
    // Destroy existing chart if it exists
    if (historyChartInstance) {
        historyChartInstance.destroy();
    }
    
    // Create new chart
    historyChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Gold Price (USD)',
                    data: prices,
                    borderColor: '#ffc107',
                    backgroundColor: 'rgba(255, 193, 7, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1,
                    pointRadius: 1,
                    pointHoverRadius: 5
                },
                {
                    label: '7-Day Moving Average',
                    data: movingAvg,
                    borderColor: '#17a2b8',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += new Intl.NumberFormat('en-US', {
                                    style: 'currency',
                                    currency: 'USD'
                                }).format(context.parsed.y);
                            }
                            return label;
                        }
                    }
                },
                legend: {
                    position: 'top',
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        tooltipFormat: 'MMM d, yyyy',
                        displayFormats: {
                            day: 'MMM d'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Date'
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Price (USD)'
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            }
        }
    });
}

// Update the history table with paginated data
function updateHistoryTable() {
    const tableBody = document.getElementById('history-table-body');
    const start = (currentPage - 1) * pageSize;
    const end = Math.min(start + pageSize, historyData.length);
    const pageData = historyData.slice(start, end);
    
    // Update pagination info
    document.getElementById('pagination-start').textContent = historyData.length > 0 ? start + 1 : 0;
    document.getElementById('pagination-end').textContent = end;
    document.getElementById('pagination-total').textContent = historyData.length;
    
    // Update pagination buttons
    document.getElementById('prev-page').disabled = currentPage === 1;
    document.getElementById('next-page').disabled = end >= historyData.length;
    
    // Clear existing rows
    tableBody.innerHTML = '';
    
    if (pageData.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="6" class="text-center">No data available</td>';
        tableBody.appendChild(row);
        return;
    }
    
    // Add data rows
    pageData.forEach(item => {
        const row = document.createElement('tr');
        
        // Format change value and class
        let changeText = '--';
        let changeClass = '';
        
        if (item.ch !== null && item.ch !== undefined) {
            const changeValue = item.ch.toFixed(2);
            const changePercent = item.chp ? item.chp.toFixed(2) : '0.00';
            const isPositive = item.ch >= 0;
            
            changeText = `${isPositive ? '+' : ''}${changeValue} (${isPositive ? '+' : ''}${changePercent}%)`;
            changeClass = isPositive ? 'text-success' : 'text-danger';
        }
        
        row.innerHTML = `
            <td>${new Date(item.date).toLocaleDateString()}</td>
            <td>$${item.price ? item.price.toFixed(2) : '--'}</td>
            <td>$${item.open ? item.open.toFixed(2) : '--'}</td>
            <td>$${item.high ? item.high.toFixed(2) : '--'}</td>
            <td>$${item.low ? item.low.toFixed(2) : '--'}</td>
            <td class="${changeClass}">${changeText}</td>
        `;
        
        tableBody.appendChild(row);
    });
}

// Fetch predictions from the API
async function fetchPredictions() {
    try {
        const model = document.getElementById('model-select').value;
        const days = document.getElementById('days-select').value;
        const predictionSummary = document.getElementById('prediction-summary');
        
        // Show loading state
        predictionSummary.innerHTML = `
            <div class="d-flex justify-content-center align-items-center py-3">
                <div class="spinner-border text-warning" role="status">
                    <span class="visually-hidden">Loading predictions...</span>
                </div>
                <span class="ms-2">Loading predictions...</span>
            </div>
        `;
        
        const response = await fetch(`${API_BASE_URL}/predict?model=${model}&days=${days}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            // Update prediction chart
            updatePredictionChart(result.predictions, model);
            
            // Update prediction summary
            updatePredictionSummary(result.predictions, model);
        } else {
            throw new Error('Failed to fetch predictions');
        }
    } catch (error) {
        console.error('Error fetching predictions:', error);
        document.getElementById('prediction-summary').innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-triangle-fill me-2"></i>
                Failed to fetch predictions. Please try again later.
            </div>
        `;
        showToast('Error', 'Failed to fetch gold price predictions. Please try again later.', 'error');
    }
}

// Update the prediction chart with data
function updatePredictionChart(predictions, modelName) {
    const ctx = document.getElementById('predictionChart').getContext('2d');
    
    // Sort predictions by date
    const sortedPredictions = [...predictions].sort((a, b) => new Date(a.date) - new Date(b.date));
    
    // Prepare data for chart
    const labels = sortedPredictions.map(item => item.date);
    const prices = sortedPredictions.map(item => item.price);
    
    // Check if we have min/max values (ensemble model)
    const hasRange = sortedPredictions[0] && 
                    sortedPredictions[0].hasOwnProperty('min_price') && 
                    sortedPredictions[0].hasOwnProperty('max_price');
    
    // Get the latest actual price for reference line
    let latestActualPrice = null;
    if (historyData && historyData.length > 0) {
        latestActualPrice = historyData[historyData.length - 1].price;
    }
    
    // Destroy existing chart if it exists
    if (predictionChartInstance) {
        predictionChartInstance.destroy();
    }
    
    // Create datasets
    const datasets = [
        {
            label: 'Predicted Price (USD)',
            data: prices,
            borderColor: '#28a745',
            backgroundColor: 'rgba(40, 167, 69, 0.1)',
            borderWidth: 2,
            fill: !hasRange,
            tension: 0.1,
            pointRadius: 4,
            pointHoverRadius: 6
        }
    ];
    
    // Add range dataset if available
    if (hasRange) {
        // Create range area
        datasets.push({
            label: 'Prediction Range',
            data: sortedPredictions.map(item => ({
                x: item.date,
                y: item.min_price,
                y1: item.max_price
            })),
            backgroundColor: 'rgba(40, 167, 69, 0.2)',
            borderWidth: 0,
            fill: '+1',
            tension: 0.1,
            pointRadius: 0,
            pointHoverRadius: 0
        });
        
        // Add min line
        datasets.push({
            label: 'Min Price',
            data: sortedPredictions.map(item => item.min_price),
            borderColor: 'rgba(40, 167, 69, 0.5)',
            borderWidth: 1,
            borderDash: [5, 5],
            fill: false,
            tension: 0.1,
            pointRadius: 0,
            pointHoverRadius: 0
        });
        
        // Add max line
        datasets.push({
            label: 'Max Price',
            data: sortedPredictions.map(item => item.max_price),
            borderColor: 'rgba(40, 167, 69, 0.5)',
            borderWidth: 1,
            borderDash: [5, 5],
            fill: false,
            tension: 0.1,
            pointRadius: 0,
            pointHoverRadius: 0
        });
    }
    
    // Create new chart
    predictionChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += new Intl.NumberFormat('en-US', {
                                    style: 'currency',
                                    currency: 'USD'
                                }).format(context.parsed.y);
                            }
                            return label;
                        }
                    }
                },
                legend: {
                    display: hasRange,
                    position: 'top',
                },
                annotation: {
                    annotations: {
                        line1: {
                            type: 'line',
                            yMin: latestActualPrice,
                            yMax: latestActualPrice,
                            borderColor: '#ffc107',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            label: {
                                content: 'Latest Actual Price',
                                display: true,
                                position: 'start',
                                backgroundColor: 'rgba(255, 193, 7, 0.8)',
                                color: '#000'
                            }
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        tooltipFormat: 'MMM d, yyyy',
                        displayFormats: {
                            day: 'MMM d'
                        }
                    },
                    title: {
                        display: true,
                        text: 'Date'
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Price (USD)'
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            }
        }
    });
}

// Update the prediction summary
function updatePredictionSummary(predictions, modelName) {
    const predictionSummary = document.getElementById('prediction-summary');
    
    if (!predictions || predictions.length === 0) {
        predictionSummary.innerHTML = `
            <div class="alert alert-warning">
                <i class="bi bi-exclamation-triangle-fill me-2"></i>
                No predictions available for the selected model and time horizon.
            </div>
        `;
        return;
    }
    
    // Sort predictions by date
    const sortedPredictions = [...predictions].sort((a, b) => new Date(a.date) - new Date(b.date));
    
    // Get first and last prediction
    const firstPrediction = sortedPredictions[0];
    const lastPrediction = sortedPredictions[sortedPredictions.length - 1];
    
    // Get latest actual price
    let latestActualPrice = null;
    let latestActualDate = null;
    if (historyData && historyData.length > 0) {
        const latest = historyData[historyData.length - 1];
        latestActualPrice = latest.price;
        latestActualDate = new Date(latest.date).toLocaleDateString();
    }
    
    // Calculate change from latest actual price to first prediction
    let changeValue = 0;
    let changePercent = 0;
    let changeClass = '';
    let changeIcon = '';
    
    if (latestActualPrice && firstPrediction) {
        changeValue = firstPrediction.price - latestActualPrice;
        changePercent = (changeValue / latestActualPrice) * 100;
        changeClass = changeValue >= 0 ? 'text-success' : 'text-danger';
        changeIcon = changeValue >= 0 ? 'bi-arrow-up-right' : 'bi-arrow-down-right';
    }
    
    // Format model name for display
    let displayModelName = modelName;
    switch (modelName) {
        case 'lstm':
            displayModelName = 'LSTM Neural Network';
            break;
        case 'linear_regression':
            displayModelName = 'Linear Regression';
            break;
        case 'random_forest':
            displayModelName = 'Random Forest';
            break;
        case 'prophet':
            displayModelName = 'Prophet';
            break;
        case 'ensemble':
            displayModelName = 'Ensemble (All Models)';
            break;
    }
    
    // Create summary HTML
    let summaryHTML = `
        <div class="card">
            <div class="card-body">
                <h6 class="card-subtitle mb-3 text-muted">Prediction Summary using ${displayModelName}</h6>
                <div class="row">
                    <div class="col-md-6">
                        <p><strong>Latest Actual Price:</strong> $${latestActualPrice ? latestActualPrice.toFixed(2) : '--'} (${latestActualDate || '--'})</p>
                        <p><strong>Next Day Prediction:</strong> $${firstPrediction.price.toFixed(2)} (${new Date(firstPrediction.date).toLocaleDateString()})</p>
                        <p><strong>Expected Change:</strong> 
                            <span class="${changeClass}">
                                <i class="bi ${changeIcon}"></i>
                                ${changeValue >= 0 ? '+' : ''}${changeValue.toFixed(2)} (${changeValue >= 0 ? '+' : ''}${changePercent.toFixed(2)}%)
                            </span>
                        </p>
                    </div>
                    <div class="col-md-6">
                        <p><strong>Prediction Horizon:</strong> ${predictions.length} days</p>
                        <p><strong>Final Prediction:</strong> $${lastPrediction.price.toFixed(2)} (${new Date(lastPrediction.date).toLocaleDateString()})</p>
    `;
    
    // Add range information if available
    if (firstPrediction.hasOwnProperty('min_price') && firstPrediction.hasOwnProperty('max_price')) {
        summaryHTML += `
                        <p><strong>Prediction Range:</strong> $${firstPrediction.min_price.toFixed(2)} - $${firstPrediction.max_price.toFixed(2)}</p>
        `;
    }
    
    summaryHTML += `
                    </div>
                </div>
            </div>
        </div>
    `;
    
    predictionSummary.innerHTML = summaryHTML;
}

// Fetch information about trained models
async function fetchModelsInfo() {
    try {
        const modelsContainer = document.getElementById('models-container');
        
        const response = await fetch(`${API_BASE_URL}/models`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            updateModelsDisplay(result.models);
        } else {
            throw new Error('Failed to fetch models information');
        }
    } catch (error) {
        console.error('Error fetching models information:', error);
        document.getElementById('models-container').innerHTML = `
            <div class="col-12">
                <div class="alert alert-danger">
                    <i class="bi bi-exclamation-triangle-fill me-2"></i>
                    Failed to fetch models information. Please try again later.
                </div>
            </div>
        `;
    }
}

// Update the models display
function updateModelsDisplay(models) {
    const modelsContainer = document.getElementById('models-container');
    
    if (!models || Object.keys(models).length === 0) {
        modelsContainer.innerHTML = `
            <div class="col-12">
                <div class="alert alert-warning">
                    <i class="bi bi-exclamation-triangle-fill me-2"></i>
                    No trained models available. Click the "Train Models" button to train prediction models.
                </div>
            </div>
        `;
        return;
    }
    
    // Clear container
    modelsContainer.innerHTML = '';
    
    // Model display names and icons
    const modelDisplayInfo = {
        'lstm': { name: 'LSTM Neural Network', icon: 'bi-braces-asterisk' },
        'linear_regression': { name: 'Linear Regression', icon: 'bi-graph-up' },
        'random_forest': { name: 'Random Forest', icon: 'bi-diagram-3' },
        'prophet': { name: 'Prophet', icon: 'bi-calendar-check' },
        'ensemble': { name: 'Ensemble Model', icon: 'bi-layers' }
    };
    
    // Add model cards
    Object.entries(models).forEach(([modelName, modelInfo]) => {
        const displayInfo = modelDisplayInfo[modelName] || { name: modelName, icon: 'bi-cpu' };
        const metrics = modelInfo.metrics || {};
        
        const modelCard = document.createElement('div');
        modelCard.className = 'col-md-6 col-lg-3 mb-4';
        modelCard.innerHTML = `
            <div class="card model-card h-100">
                <div class="card-header d-flex align-items-center">
                    <i class="bi ${displayInfo.icon} me-2"></i>
                    ${displayInfo.name}
                </div>
                <div class="card-body">
                    <h6 class="card-subtitle mb-3 text-muted">Performance Metrics</h6>
                    <div class="model-metric">
                        <span>MSE:</span>
                        <span class="model-metric-value">${metrics.mse ? metrics.mse.toFixed(2) : 'N/A'}</span>
                    </div>
                    <div class="model-metric">
                        <span>RMSE:</span>
                        <span class="model-metric-value">${metrics.rmse ? metrics.rmse.toFixed(2) : 'N/A'}</span>
                    </div>
                    <div class="model-metric">
                        <span>MAE:</span>
                        <span class="model-metric-value">${metrics.mae ? metrics.mae.toFixed(2) : 'N/A'}</span>
                    </div>
                    <div class="model-metric">
                        <span>R²:</span>
                        <span class="model-metric-value">${metrics.r2 ? metrics.r2.toFixed(4) : 'N/A'}</span>
                    </div>
                </div>
                <div class="card-footer text-center">
                    <button class="btn btn-sm btn-outline-primary use-model-btn" data-model="${modelName}">
                        Use for Prediction
                    </button>
                </div>
            </div>
        `;
        
        modelsContainer.appendChild(modelCard);
    });
    
    // Add event listeners to "Use for Prediction" buttons
    document.querySelectorAll('.use-model-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const modelName = e.target.dataset.model;
            document.getElementById('model-select').value = modelName;
            
            // Scroll to predictions section
            document.querySelector('#predictions').scrollIntoView({
                behavior: 'smooth'
            });
            
            // Fetch predictions with selected model
            fetchPredictions();
        });
    });
}

// Train prediction models
async function trainModels() {
    try {
        const trainBtn = document.getElementById('train-btn');
        
        // Disable button and show loading state
        trainBtn.disabled = true;
        trainBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Training...';
        
        showToast('Training Models', 'Model training has started. This may take a few minutes...', 'info');
        
        const response = await fetch(`${API_BASE_URL}/train`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            showToast('Training Complete', 'Models have been successfully trained and are ready for predictions.', 'success');
            
            // Refresh models info
            await fetchModelsInfo();
            
            // Refresh predictions
            await fetchPredictions();
        } else {
            throw new Error('Failed to train models');
        }
    } catch (error) {
        console.error('Error training models:', error);
        showToast('Training Failed', 'Failed to train prediction models. Please try again later.', 'error');
    } finally {
        // Reset button state
        const trainBtn = document.getElementById('train-btn');
        trainBtn.disabled = false;
        trainBtn.innerHTML = '<i class="bi bi-gear-fill me-1"></i> Train Models';
    }
}

// Refresh all data
async function refreshAllData() {
    try {
        const refreshBtn = document.getElementById('refresh-btn');
        
        // Disable button and show loading state
        refreshBtn.disabled = true;
        refreshBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Refreshing...';
        
        // Fetch new data
        await Promise.all([
            fetchCurrentPrice(),
            fetchHistoricalData(30),  // Refresh with default 30 days
            fetchModelsInfo()
        ]);
        
        // Refresh predictions with current settings
        await fetchPredictions();
        
        showToast('Data Refreshed', 'All gold price data has been refreshed successfully.', 'success');
    } catch (error) {
        console.error('Error refreshing data:', error);
        showToast('Refresh Failed', 'Failed to refresh data. Please try again later.', 'error');
    } finally {
        // Reset button state
        const refreshBtn = document.getElementById('refresh-btn');
        refreshBtn.disabled = false;
        refreshBtn.innerHTML = '<i class="bi bi-arrow-clockwise me-1"></i> Refresh Data';
    }
}

// Show toast notification
function showToast(title, message, type = 'info') {
    const toastEl = document.getElementById('toast');
    const toastTitle = document.getElementById('toast-title');
    const toastMessage = document.getElementById('toast-message');
    
    // Set icon based on type
    let icon = 'bi-info-circle';
    switch (type) {
        case 'success':
            icon = 'bi-check-circle';
            break;
        case 'error':
            icon = 'bi-exclamation-circle';
            break;
        case 'warning':
            icon = 'bi-exclamation-triangle';
            break;
    }
    
    // Set title and message
    toastTitle.innerHTML = `<i class="bi ${icon} me-2"></i>${title}`;
    toastMessage.textContent = message;
    
    // Set toast class based on type
    toastEl.className = 'toast';
    toastEl.classList.add(`border-${type === 'info' ? 'primary' : type}`);
    
    // Show the toast
    const toastInstance = new bootstrap.Toast(toastEl);
    toastInstance.show();
}
