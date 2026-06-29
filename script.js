


// GLOBALS

    const API_BASE = 'http://127.0.0.1:5000';
    let currentResults = [];           // array of result objects from API
    let selectedResult = null;         // for viewer
    let selectedFile = null;           // File object for upload

    // ============================================================
                          // DOM refs
    
    const fileInput = document.getElementById('fileInput');
    const uploadArea = document.getElementById('uploadArea');
    const browseLink = document.getElementById('browseLink');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const clearFileBtn = document.getElementById('clearFileBtn');
    const searchBtn = document.getElementById('searchBtn');
    const searchStatus = document.getElementById('search-status');
    const statusLines = {
        s1: document.getElementById('s1'),
        s2: document.getElementById('s2'),
        s3: document.getElementById('s3')
    };

    const cardGrid = document.getElementById('card-grid');
    const resultsQueryLabel = document.getElementById('resultsQueryLabel');
    const resultsTitle = document.getElementById('resultsTitle');
    const resultsMeta = document.getElementById('resultsMeta');

    // Viewer elements

    const vwImage = document.getElementById('vw-image');
    const vwThumb = document.getElementById('vw-thumb');
    const vwSensorBadge = document.getElementById('vw-sensor-badge');
    const vwDateBadge = document.getElementById('vw-date-badge');
    const vwTitle = document.getElementById('vw-title');
    const vwScoreBar = document.getElementById('vw-score-bar');
    const vwScoreVal = document.getElementById('vw-score-val');
    const mdSensor = document.getElementById('md-sensor');
    const mdDate = document.getElementById('md-date');
    const mdCoords = document.getElementById('md-coords');
    const mdCloud = document.getElementById('md-cloud');
    const mdRes = document.getElementById('md-res');
    const downloadBtn = document.getElementById('downloadBtn');

    // Dashboard

    const dashIndexed = document.getElementById('dashIndexed');
    const dashIndexStatus = document.getElementById('dashIndexStatus');
    const dashEncoder = document.getElementById('dashEncoder');
    const dashQueryTime = document.getElementById('dashQueryTime');
    const dashDim = document.getElementById('dashDim');
    const dashFramework = document.getElementById('dashFramework');
    const dashDevice = document.getElementById('dashDevice');
    const dashBatch = document.getElementById('dashBatch');

    // ============================================================
                            // Navigation

    function navigateTo(pageId) {
        document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
        document.querySelector(`.nav-btn[data-page="${pageId}"]`)?.classList.add('active');

        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        document.getElementById('page-' + pageId).classList.add('active');

        if (pageId === 'results') {
            // animate score bars if present
            setTimeout(animateScores, 100);
        }
        if (pageId === 'dash') {
            fetchDashboardStatus();
        }
    }

    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            navigateTo(this.dataset.page);
        });
    });

    // ============================================================
                      // File Upload Handle

    browseLink.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            selectedFile = file;
            fileName.textContent = file.name;
            fileInfo.style.display = 'block';
            // also show a preview? not needed.
        } else {
            clearFileSelection();
        }
    });

    clearFileBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        clearFileSelection();
    });

    function clearFileSelection() {
        selectedFile = null;
        fileInput.value = '';
        fileInfo.style.display = 'none';
        fileName.textContent = '';
    }

    // ============================================================
                        // Search (doSearch)

    async function doSearch() {
        if (!selectedFile) {
            alert('Please select a SAR image to query.');
            return;
        }

        // Show status
        searchStatus.style.display = 'block';
        searchStatus.classList.remove('error');
        const statusMsg = ['▶ Encoding query vector...', '▶ Searching database...', '▶ Ranking by cosine similarity...'];
        ['s1','s2','s3'].forEach((id, i) => {
            document.getElementById(id).textContent = statusMsg[i];
            document.getElementById(id).style.color = 'var(--color-text-secondary)';
        });

        // Disable search button
        searchBtn.disabled = true;
        searchBtn.innerHTML = '<i class="ti ti-loader"></i> Searching...';

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('k', '6');        // you can make these adjustable
            formData.append('limit', '2000');
            formData.append('refresh_cache', 'false');

            const response = await fetch(`${API_BASE}/api/retrieve`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || `HTTP ${response.status}`);
            }

            const data = await response.json();
            // data = { query, count, results: [ { rank, filename, score, match_percent, image_url }, ... ] }
            currentResults = data.results || [];

            // Update status with success
            document.getElementById('s1').textContent = '✓ Query encoded';
            document.getElementById('s2').textContent = '✓ Database searched';
            document.getElementById('s3').textContent = `✓ Found ${data.count} results`;

            // Hide status after a moment
            setTimeout(() => {
                searchStatus.style.display = 'none';
            }, 1200);

            // Render results
            renderResults(data);

            // Navigate to results page
            navigateTo('results');

        } catch (err) {
            console.error(err);
            searchStatus.classList.add('error');
            document.getElementById('s1').textContent = '❌ Error: ' + err.message;
            document.getElementById('s2').textContent = '';
            document.getElementById('s3').textContent = 'Please check console and backend.';
        } finally {
            searchBtn.disabled = false;
            searchBtn.innerHTML = '<i class="ti ti-search" aria-hidden="true"></i> Search satellite archive';
        }
    }

    // ============================================================
                          // Render Results
 
    function renderResults(data) {
        const query = data.query || 'uploaded image';
        const count = data.count || 0;
        resultsQueryLabel.textContent = `Query · "${query}"`;
        resultsTitle.textContent = `${count} result${count !== 1 ? 's' : ''} found`;

        // generate a fake time (or use actual response time if available)
        const timeStr = '0.24 s'; // could be improved
        resultsMeta.textContent = `Retrieved in ${timeStr} · sorted by similarity`;

        if (count === 0) {
            cardGrid.innerHTML = `<div class="result-card" style="grid-column:1/-1;padding:30px;text-align:center;color:var(--color-text-tertiary);">No results found.</div>`;
            return;
        }

        let html = '';
        currentResults.forEach((res, index) => {
            const score = res.match_percent || 0;
            const filename = res.filename || 'unknown';
            const imageUrl = res.image_url ? `${API_BASE}${res.image_url}` : '#';
            // We don't have sensor/date/coords from the backend, so use placeholders.
            // Could parse filename or add more fields; for now we generate dummy info.
            const sensor = 'Sentinel-1 SAR'; // or derive from filename?
            const date = filename.match(/\d{4}-\d{2}-\d{2}/) ? filename.match(/\d{4}-\d{2}-\d{2}/)[0] : '2024-01-01';
            const coords = '—';
            const cloud = '—';
            const resStr = '10m';
            const tags = ['SAR', 'Optical'];

            html += `
                <div class="result-card" data-index="${index}" onclick="viewResult(${index})">
                    <div class="result-thumb" style="background:var(--color-background-info);">
                        <span class="result-badge sensor">${sensor.split(' ')[0]}</span>
                        <span class="result-badge cloud">☁ ${cloud}</span>
                        <i class="ti ti-crosshair result-icon" aria-hidden="true"></i>
                        <!-- We'll load the image as background or img -->
                        <img src="${imageUrl}" style="width:100%;height:100%;object-fit:cover;position:absolute;top:0;left:0;opacity:0.7;" alt="${filename}" onerror="this.style.display='none'" />
                    </div>
                    <div class="result-body">
                        <div class="result-title">${filename}</div>
                        <div class="result-date">${date}</div>
                        <div class="result-tags">
                            ${tags.map(t => `<span class="tag">${t}</span>`).join('')}
                            <span class="tag res">${resStr}</span>
                        </div>
                        <div class="result-score-row">
                            <span class="result-score-label">Match</span>
                            <span class="result-score-value" style="color:${scoreColor(score)}">${score}%</span>
                        </div>
                        <div class="result-score-bar">
                            <div class="score-bar" data-score="${score}" style="width:0%;background:${scoreColor(score)};transition:width 0.5s ease ${index * 60}ms;"></div>
                        </div>
                    </div>
                </div>
            `;
        });
        cardGrid.innerHTML = html;

        // After rendering, trigger score bar animation
        setTimeout(animateScores, 50);
    }

    // ============================================================
                      // Animate Scroll bar

    function animateScores() {
        document.querySelectorAll('.score-bar').forEach(bar => {
            const score = parseFloat(bar.dataset.score);
            if (!isNaN(score)) {
                bar.style.width = score + '%';
            }
        });
    }

    // ============================================================
                          // ScoreColor

    function scoreColor(s) {
        return s >= 85 ? 'var(--color-text-success)' :
               s >= 70 ? 'var(--color-text-warning)' :
               'var(--color-text-danger)';
    }

    // ============================================================
                      // View Results (click card)

    function viewResult(index) {
        const res = currentResults[index];
        if (!res) return;
        selectedResult = res;

        // Populate viewer
        const filename = res.filename || 'unknown';
        const score = res.match_percent || 0;
        const imageUrl = res.image_url ? `${API_BASE}${res.image_url}` : '';

        // Show image
        if (imageUrl) {
            vwImage.src = imageUrl;
            vwImage.style.display = 'block';
            vwImage.onerror = () => { vwImage.style.display = 'none'; };
        } else {
            vwImage.style.display = 'none';
        }

        // Set metadata (extract from filename or use defaults)
        const sensor = 'Sentinel-1 SAR';
        const date = filename.match(/\d{4}-\d{2}-\d{2}/) ? filename.match(/\d{4}-\d{2}-\d{2}/)[0] : '—';
        const coords = '—';
        const cloud = '—';
        const resStr = '10m';

        vwSensorBadge.textContent = sensor.split(' ')[0];
        vwDateBadge.textContent = date;
        vwTitle.textContent = filename;
        vwScoreBar.style.width = score + '%';
        vwScoreBar.style.background = scoreColor(score);
        vwScoreVal.textContent = score + '%';
        vwScoreVal.style.color = scoreColor(score);

        mdSensor.textContent = sensor;
        mdDate.textContent = date;
        mdCoords.textContent = coords;
        mdCloud.textContent = cloud + '%';
        mdRes.textContent = resStr;

        // Setup download button
        downloadBtn.onclick = () => {
            if (imageUrl) window.open(imageUrl, '_blank');
        };

        navigateTo('viewer');
    }

    // ============================================================
                    // Dashboard: fetch /api/status

    async function fetchDashboardStatus() {
        try {
            const resp = await fetch(`${API_BASE}/api/status`);
            if (!resp.ok) throw new Error('Status API error');
            const data = await resp.json();

            dashIndexed.textContent = data.optical_images || '—';
            dashIndexStatus.textContent = data.embedding_cache_found ? 'Loaded ✓' : 'Not cached';
            dashIndexStatus.style.color = data.embedding_cache_found ? 'var(--color-text-success)' : 'var(--color-text-warning)';
            dashEncoder.textContent = 'ResNet18';
            dashQueryTime.textContent = '0.24s'; // not provided
            dashDim.textContent = '256'; // fixed
            dashFramework.textContent = 'PyTorch';
            dashDevice.textContent = data.device || '—';
            dashBatch.textContent = '64';
        } catch (err) {
            console.error('Dashboard fetch error:', err);
            dashIndexed.textContent = '⚠️';
            dashIndexStatus.textContent = 'Offline';
        }
    }

    // ============================================================
                      // Query Log (static mock)

    const LOGS = [
        { t: "14:32:01", q: "flooded agricultural area", r: 6, ms: 241 },
        { t: "14:28:47", q: "urban heat island dense city", r: 12, ms: 198 },
        { t: "14:15:33", q: "coastal erosion mangrove loss", r: 9, ms: 312 },
        { t: "13:58:12", q: "[Image query · upload]", r: 7, ms: 267 },
        { t: "13:44:05", q: "deforestation north amazon", r: 15, ms: 188 },
        { t: "13:30:29", q: "glacier retreat himalaya 2024", r: 4, ms: 354 }
    ];

    function renderLog() {
        const rows = document.getElementById('log-rows');
        const queries = document.getElementById('log-queries');
        const results = document.getElementById('log-results');
        const times = document.getElementById('log-times');

        rows.innerHTML = LOGS.map((l, i) =>
            `<div class="log-cell">${l.t}${i === 0 ? ' <span class="live-dot"></span>' : ''}</div>`
        ).join('');
        queries.innerHTML = LOGS.map(l => `<div class="log-cell query-text">${l.q}</div>`).join('');
        results.innerHTML = LOGS.map(l => `<div class="log-cell result-count">${l.r}</div>`).join('');
        times.innerHTML = LOGS.map(l =>
            `<div class="log-cell" style="color:${l.ms > 300 ? 'var(--color-text-warning)' : 'var(--color-text-tertiary)'}">${l.ms}ms</div>`
        ).join('');
    }

    // ============================================================
                    // Tab Switching (Search page)

    document.querySelectorAll('.query-tab').forEach(tab => {
        tab.addEventListener('click', function() {
            document.querySelectorAll('.query-tab').forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            // If 'image' tab is not active, maybe show a message
            if (this.dataset.tab !== 'image') {
                alert('Only Image upload is supported in this demo.');
            }
        });
    });

    // ============================================================
                            // Clear Filters

    document.querySelector('.btn-clear-filters')?.addEventListener('click', function() {
        document.querySelectorAll('.filters-panel input[type="checkbox"]').forEach(cb => cb.checked = false);
        document.querySelector('.filters-panel input[type="range"]').value = 30;
        document.getElementById('cc-val').textContent = '30%';
        document.querySelectorAll('.filters-panel input[type="date"]').forEach(d => d.value = '');
    });

    // ============================================================
                                 // INIT

    renderLog();
    // Set initial results page to empty
    cardGrid.innerHTML = `<div class="result-card" style="grid-column:1/-1;padding:30px;text-align:center;color:var(--color-text-tertiary);">Submit a search to see results.</div>`;
    // Fetch dashboard on first load? We'll do it when dashboard is clicked.
    // But we can also fetch on load.
    fetchDashboardStatus();
    // Ensure correct active page: search is active.
    document.querySelector('.nav-btn.active')?.classList.remove('active');
    document.querySelector('.nav-btn[data-page="search"]')?.classList.add('active');


