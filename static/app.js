/**
 * RAG Teaching Assistant - Interactive Web UI Frontend Logic
 * Supports Automatic Session Data Isolation
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM References
  const userSessionBar = document.getElementById('userSessionBar');
  const sessionUsername = document.getElementById('sessionUsername');
  const newSessionBtn = document.getElementById('newSessionBtn');

  const navItems = document.querySelectorAll('.nav-item');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const pageTitle = document.getElementById('pageTitle');
  const pageSubtitle = document.getElementById('pageSubtitle');
  
  const askForm = document.getElementById('askForm');
  const questionInput = document.getElementById('questionInput');
  const chatMessages = document.getElementById('chatMessages');
  const citationsList = document.getElementById('citationsList');
  const citationCountBadge = document.getElementById('citationCountBadge');
  const videoFilterSelect = document.getElementById('videoFilterSelect');
  const quickPrompts = document.getElementById('quickPrompts');

  const mediaPreviewContainer = document.getElementById('mediaPreviewContainer');
  const videoPlayer = document.getElementById('videoPlayer');
  const playerInfo = document.getElementById('playerInfo');

  const videoFileInput = document.getElementById('videoFileInput');
  const dropzone = document.getElementById('dropzone');
  const pipelineProgress = document.getElementById('pipelineProgress');
  const pipelineProgressFill = document.getElementById('pipelineProgressFill');
  const pipelineStatusText = document.getElementById('pipelineStatusText');
  const pipelinePercentText = document.getElementById('pipelinePercentText');
  const ingestResultCard = document.getElementById('ingestResultCard');
  const resultTitle = document.getElementById('resultTitle');
  const resultDetails = document.getElementById('resultDetails');

  const videoGrid = document.getElementById('videoGrid');
  const refreshLibraryBtn = document.getElementById('refreshLibraryBtn');
  const transcriptVideoList = document.getElementById('transcriptVideoList');
  const segmentsList = document.getElementById('segmentsList');
  const transcriptTitle = document.getElementById('transcriptTitle');
  const transcriptSubtitle = document.getElementById('transcriptSubtitle');
  const transcriptSearchInput = document.getElementById('transcriptSearchInput');

  let currentVideos = [];
  let activeTranscriptSegments = [];

  // Get or Generate Private Session ID for Data Isolation
  function getSessionId() {
    let sessId = localStorage.getItem('rag_session_id');
    if (!sessId) {
      sessId = 'sess_' + Math.random().toString(36).substring(2, 10) + Date.now().toString(36).substring(4);
      localStorage.setItem('rag_session_id', sessId);
    }
    return sessId;
  }

  function getSessionHeader() {
    return { 'X-Session-ID': getSessionId() };
  }

  async function fetchWithSession(url, options = {}) {
    options.headers = {
      ...options.headers,
      ...getSessionHeader()
    };
    return await fetch(url, options);
  }

  // Update session badge text
  const currentSessId = getSessionId();
  if (sessionUsername) {
    sessionUsername.textContent = 'Private: ' + currentSessId.substring(0, 12);
  }

  // Handle New Private Session Creation
  if (newSessionBtn) {
    newSessionBtn.addEventListener('click', () => {
      if (confirm('Start a fresh, private workspace? Your current workspace data remains isolated in its own private session.')) {
        const freshId = 'sess_' + Math.random().toString(36).substring(2, 10) + Date.now().toString(36).substring(4);
        localStorage.setItem('rag_session_id', freshId);
        sessionUsername.textContent = 'Private: ' + freshId.substring(0, 12);
        loadVideoOptions();
        loadVideoLibrary();
      }
    });
  }

  // Mobile Sidebar Drawer Controls
  const sidebar = document.getElementById('sidebar');
  const mobileMenuBtn = document.getElementById('mobileMenuBtn');
  const sidebarCloseBtn = document.getElementById('sidebarCloseBtn');
  const sidebarOverlay = document.getElementById('sidebarOverlay');

  function openSidebar() {
    if (sidebar) sidebar.classList.add('open');
    if (sidebarOverlay) sidebarOverlay.classList.add('active');
  }

  function closeSidebar() {
    if (sidebar) sidebar.classList.remove('open');
    if (sidebarOverlay) sidebarOverlay.classList.remove('active');
  }

  if (mobileMenuBtn) mobileMenuBtn.addEventListener('click', openSidebar);
  if (sidebarCloseBtn) sidebarCloseBtn.addEventListener('click', closeSidebar);
  if (sidebarOverlay) sidebarOverlay.addEventListener('click', closeSidebar);

  // Tab Navigation
  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const tabId = item.getAttribute('data-tab');
      
      navItems.forEach(n => n.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      item.classList.add('active');
      const targetPane = document.getElementById(tabId);
      if (targetPane) targetPane.classList.add('active');

      // Close mobile drawer after selecting a tab
      if (window.innerWidth <= 992) {
        closeSidebar();
      }

      switch (tabId) {
        case 'chat-tab':
          pageTitle.textContent = 'Lecture Video Q&A Assistant';
          pageSubtitle.textContent = 'Ask questions grounded in transcript excerpts with exact video timestamps.';
          break;
        case 'ingest-tab':
          pageTitle.textContent = 'Ingest New Lecture Video';
          pageSubtitle.textContent = 'Upload lecture MP4/audio files to transcribe with Whisper and index into FAISS.';
          break;
        case 'library-tab':
          pageTitle.textContent = 'Ingested Lecture Library';
          pageSubtitle.textContent = 'View all indexed lecture videos in your private session.';
          loadVideoLibrary();
          break;
        case 'transcript-tab':
          pageTitle.textContent = 'Transcript Browser';
          pageSubtitle.textContent = 'Browse and search raw timestamped transcript segments.';
          loadTranscriptBrowser();
          break;
      }
    });
  });

  // Fetch Health & System Status
  async function fetchHealthStatus() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        const data = await res.json();
        document.getElementById('statusDot').className = 'status-indicator online';
        document.getElementById('statusText').textContent = 'System Active';
        document.getElementById('statusProvider').textContent = (data.provider || 'Gemini').toUpperCase();
        document.getElementById('statusWhisper').textContent = (data.whisper_model || 'Base').toUpperCase();
        document.getElementById('statusEmbed').textContent = 'MiniLM-L6-v2';
      }
    } catch (err) {
      document.getElementById('statusDot').className = 'status-indicator';
      document.getElementById('statusText').textContent = 'Backend Offline';
    }
  }
  fetchHealthStatus();

  // Load Ingested Video Select Filter Options
  async function loadVideoOptions() {
    try {
      const res = await fetchWithSession('/videos');
      if (res && res.ok) {
        currentVideos = await res.json();
        videoFilterSelect.innerHTML = '<option value="">All Lectures (Global Search)</option>';
        currentVideos.forEach(v => {
          const opt = document.createElement('option');
          opt.value = v.video_id;
          opt.textContent = `${v.source_filename} (${v.num_chunks} chunks)`;
          videoFilterSelect.appendChild(opt);
        });
        return true;
      }
      return false;
    } catch (err) {
      console.error("Error loading video options:", err);
      return false;
    }
  }
  loadVideoOptions();

  // Quick Prompt Chips
  if (quickPrompts) {
    quickPrompts.addEventListener('click', (e) => {
      const chip = e.target.closest('.prompt-chip');
      if (chip) {
        const text = chip.getAttribute('data-prompt');
        questionInput.value = text;
        askForm.dispatchEvent(new Event('submit'));
      }
    });
  }

  let qaHistory = [];
  const exportNotesBtn = document.getElementById('exportNotesBtn');

  if (exportNotesBtn) {
    exportNotesBtn.addEventListener('click', async () => {
      if (qaHistory.length === 0) {
        alert('No Q&A history yet! Ask a question first to generate study notes.');
        return;
      }

      try {
        const res = await fetchWithSession('/export-notes', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            title: 'Lecture Study Guide & Q&A Summary',
            messages: qaHistory
          })
        });

        if (!res.ok) {
          throw new Error('Failed to generate export file');
        }

        const blob = await res.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'lecture_study_notes.md';
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
      } catch (err) {
        alert(`Export Error: ${err.message}`);
      }
    });
  }

  // Q&A /ask Form Submission
  askForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    appendMessage(question, 'user');
    questionInput.value = '';

    const loadingId = appendLoadingMessage();

    try {
      const payload = {
        question: question,
        top_k: 5
      };
      if (videoFilterSelect.value) {
        payload.video_id = videoFilterSelect.value;
      }

      const res = await fetchWithSession('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      removeMessage(loadingId);

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        appendMessage(`Error: ${errorData.detail || 'Failed to generate answer'}`, 'bot');
        return;
      }

      const data = await res.json();
      appendMessage(data.answer, 'bot');
      renderCitations(data.citations);

      qaHistory.push({
        question: question,
        answer: data.answer,
        citations: data.citations
      });

    } catch (err) {
      removeMessage(loadingId);
      appendMessage(`Error connecting to RAG backend: ${err.message}`, 'bot');
    }
  });

  function appendMessage(text, sender) {
    const welcomeCard = document.querySelector('.welcome-card');
    if (welcomeCard) welcomeCard.style.display = 'none';

    const bubble = document.createElement('div');
    bubble.className = `msg-bubble ${sender}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.innerHTML = sender === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';

    const textDiv = document.createElement('div');
    textDiv.className = 'msg-text';
    textDiv.innerHTML = formatMarkdown(text);

    bubble.appendChild(avatar);
    bubble.appendChild(textDiv);
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendLoadingMessage() {
    const id = 'msg-' + Date.now();
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble bot';
    bubble.id = id;

    const avatar = document.createElement('div');
    avatar.className = 'msg-avatar';
    avatar.innerHTML = '<i class="fa-solid fa-robot"></i>';

    const textDiv = document.createElement('div');
    textDiv.className = 'msg-text';
    textDiv.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Searching vector index & generating answer...';

    bubble.appendChild(avatar);
    bubble.appendChild(textDiv);
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
  }

  function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function formatMarkdown(str) {
    if (!str) return '';
    return str
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/\*([^*]+)\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/^>\s*(.+)$/gm, '<blockquote style="border-left: 3px solid var(--accent-primary); padding-left: 10px; margin: 6px 0; color: var(--text-muted);">$1</blockquote>')
      .replace(/\n\n/g, '<br><br>')
      .replace(/\n/g, '<br>');
  }

  // Render Citation Cards
  function renderCitations(citations) {
    citationsList.innerHTML = '';
    if (!citations || citations.length === 0) {
      citationCountBadge.textContent = '0 Sources';
      citationsList.innerHTML = `
        <div class="empty-citations">
          <i class="fa-solid fa-circle-exclamation"></i>
          <p>No matching video transcript excerpts found in vector store.</p>
        </div>`;
      return;
    }

    citationCountBadge.textContent = `${citations.length} Sources`;

    citations.forEach((c) => {
      const card = document.createElement('div');
      card.className = 'citation-card';
      
      const formatTime = (sec) => {
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return `${m}:${s < 10 ? '0' : ''}${s}`;
      };

      const scorePercent = Math.round(c.score * 100);

      card.innerHTML = `
        <div class="citation-card-header">
          <span class="citation-time">
            <i class="fa-solid fa-play-circle"></i> ${formatTime(c.start)} - ${formatTime(c.end)}
          </span>
          <span class="citation-score">${scorePercent}% Match</span>
        </div>
        <div class="citation-source" title="${c.source_filename}">
          <i class="fa-solid fa-file-video"></i> ${c.source_filename}
        </div>
      `;

      card.addEventListener('click', () => {
        playCitationTimestamp(c);
      });

      citationsList.appendChild(card);
    });
  }

  function playCitationTimestamp(citation) {
    const videoObj = currentVideos.find(v => v.video_id === citation.video_id);
    if (videoObj && videoObj.media_url) {
      mediaPreviewContainer.style.display = 'block';
      videoPlayer.src = videoObj.media_url;
      videoPlayer.currentTime = citation.start;
      videoPlayer.play().catch(() => {});
      playerInfo.textContent = `Playing ${citation.source_filename} at timestamp ${Math.floor(citation.start)}s`;
    } else {
      mediaPreviewContainer.style.display = 'block';
      playerInfo.textContent = `Timestamp ${Math.floor(citation.start)}s in ${citation.source_filename}`;
    }
  }

  // Video Upload Ingestion Handling
  if (dropzone) {
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        handleVideoIngest(e.dataTransfer.files[0]);
      }
    });
  }

  if (videoFileInput) {
    videoFileInput.addEventListener('change', () => {
      if (videoFileInput.files.length > 0) {
        handleVideoIngest(videoFileInput.files[0]);
      }
    });
  }

  async function pollIngestTaskStatus(taskId) {
    let currentPct = 40;
    return new Promise((resolve, reject) => {
      const interval = setInterval(async () => {
        try {
          const res = await fetchWithSession(`/ingest/${taskId}/status`);
          if (!res.ok) return;
          const statusData = await res.json();

          if (statusData.status === 'downloading') {
            updateStep('stepUpload', 'completed');
            updateStep('stepFFmpeg', 'active');
            pipelineProgressFill.style.width = '35%';
            pipelineStatusText.textContent = statusData.message || 'Downloading audio with yt-dlp...';
            pipelinePercentText.textContent = '35%';
          } else if (statusData.status === 'transcribing') {
            updateStep('stepUpload', 'completed');
            updateStep('stepFFmpeg', 'completed');
            updateStep('stepWhisper', 'active');
            if (currentPct < 82) currentPct += 4;
            pipelineProgressFill.style.width = `${currentPct}%`;
            pipelineStatusText.textContent = statusData.message || 'Transcribing speech with Whisper...';
            pipelinePercentText.textContent = `${currentPct}%`;
          } else if (statusData.status === 'indexing') {
            updateStep('stepWhisper', 'completed');
            updateStep('stepFAISS', 'active');
            pipelineProgressFill.style.width = '92%';
            pipelineStatusText.textContent = statusData.message || 'Indexing vector embeddings into FAISS...';
            pipelinePercentText.textContent = '92%';
          } else if (statusData.status === 'completed') {
            clearInterval(interval);
            updateStep('stepFAISS', 'completed');
            pipelineProgressFill.style.width = '100%';
            pipelineStatusText.textContent = 'Ingestion complete!';
            pipelinePercentText.textContent = '100%';
            resolve(statusData.result || {});
          } else if (statusData.status === 'failed') {
            clearInterval(interval);
            reject(new Error(statusData.message || 'Async ingestion failed'));
          }
        } catch (e) {
          // Retry polling on transient network hiccup
        }
      }, 1500);
    });
  }

  async function handleVideoIngest(file) {
    dropzone.style.display = 'none';
    pipelineProgress.style.display = 'block';
    ingestResultCard.style.display = 'none';

    updateStep('stepUpload', 'active');
    pipelineProgressFill.style.width = '20%';
    pipelineStatusText.textContent = `Uploading ${file.name}...`;
    pipelinePercentText.textContent = '20%';

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetchWithSession('/ingest', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Ingestion failed');
      }

      const data = await res.json();
      const result = await pollIngestTaskStatus(data.task_id);

      setTimeout(() => {
        pipelineProgress.style.display = 'none';
        ingestResultCard.style.display = 'flex';
        resultTitle.textContent = 'Lecture Video Successfully Ingested!';
        resultDetails.textContent = `Extracted ${result.num_segments || 0} transcript segments and created ${result.num_chunks || 0} FAISS embeddings for video ID ${result.video_id || data.video_id}.`;
        dropzone.style.display = 'block';
        loadVideoOptions();
      }, 1000);

    } catch (err) {
      alert(`Ingestion Error: ${err.message}`);
      pipelineProgress.style.display = 'none';
      dropzone.style.display = 'block';
    }
  }

  // YouTube Ingestion Form Handler
  const youtubeIngestForm = document.getElementById('youtubeIngestForm');
  const youtubeUrlInput = document.getElementById('youtubeUrlInput');

  if (youtubeIngestForm) {
    youtubeIngestForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const url = youtubeUrlInput.value.trim();
      if (!url) return;

      dropzone.style.display = 'none';
      pipelineProgress.style.display = 'block';
      ingestResultCard.style.display = 'none';
      updateStep('stepUpload', 'active');
      pipelineProgressFill.style.width = '15%';
      pipelineStatusText.textContent = 'Connecting to YouTube via yt-dlp...';
      pipelinePercentText.textContent = '15%';

      try {
        const res = await fetchWithSession('/ingest-youtube', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url })
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || 'YouTube ingestion failed');
        }

        const data = await res.json();
        const result = await pollIngestTaskStatus(data.task_id);

        setTimeout(() => {
          pipelineProgress.style.display = 'none';
          ingestResultCard.style.display = 'flex';
          resultTitle.textContent = 'YouTube Lecture Video Ingested!';
          resultDetails.textContent = `Extracted ${result.num_segments || 0} transcript segments and created ${result.num_chunks || 0} FAISS embeddings.`;
          dropzone.style.display = 'block';
          youtubeUrlInput.value = '';
          loadVideoOptions();
        }, 1000);
      } catch (err) {
        alert(`YouTube Ingestion Error: ${err.message}`);
        pipelineProgress.style.display = 'none';
        dropzone.style.display = 'block';
      }
    });
  }

  function updateStep(stepId, state) {
    const el = document.getElementById(stepId);
    if (el) {
      el.className = `step-item ${state}`;
    }
  }

  // Load Library Tab
  async function loadVideoLibrary() {
    videoGrid.innerHTML = '<div class="empty-citations">Loading video library...</div>';
    await loadVideoOptions();

    if (currentVideos.length === 0) {
      videoGrid.innerHTML = `
        <div class="empty-citations" style="grid-column: 1 / -1;">
          <i class="fa-solid fa-folder-open"></i>
          <p>No lecture videos indexed in your private session yet. Go to "Ingest Lecture" to add your first video.</p>
        </div>`;
      return;
    }

    videoGrid.innerHTML = '';
    currentVideos.forEach(v => {
      const card = document.createElement('div');
      card.className = 'video-card';
      card.innerHTML = `
        <div class="video-card-title"><i class="fa-solid fa-file-video"></i> ${v.source_filename}</div>
        <div class="video-card-stats">
          <div class="stat-item"><i class="fa-solid fa-list-check"></i> ${v.num_segments} Segments</div>
          <div class="stat-item"><i class="fa-solid fa-database"></i> ${v.num_chunks} Vector Chunks</div>
        </div>
      `;
      videoGrid.appendChild(card);
    });
  }

  if (refreshLibraryBtn) {
    refreshLibraryBtn.addEventListener('click', loadVideoLibrary);
  }

  // Load Transcript Browser Tab
  async function loadTranscriptBrowser() {
    await loadVideoOptions();
    transcriptVideoList.innerHTML = '';

    if (currentVideos.length === 0) {
      transcriptVideoList.innerHTML = '<div class="empty-list">No lectures available.</div>';
      return;
    }

    currentVideos.forEach(v => {
      const item = document.createElement('div');
      item.className = 'lecture-item';
      item.textContent = v.source_filename;
      item.addEventListener('click', () => {
        document.querySelectorAll('.lecture-item').forEach(i => i.classList.remove('active'));
        item.classList.add('active');
        fetchAndDisplayTranscript(v);
      });
      transcriptVideoList.appendChild(item);
    });
  }

  async function fetchAndDisplayTranscript(videoObj) {
    transcriptTitle.textContent = videoObj.source_filename;
    transcriptSubtitle.textContent = `Language: ${videoObj.language.toUpperCase()} • ${videoObj.num_segments} Timestamped Segments`;
    segmentsList.innerHTML = '<div class="empty-segments">Loading transcript segments...</div>';

    try {
      const res = await fetchWithSession(`/videos/${videoObj.video_id}/transcript`);
      if (res && res.ok) {
        const transcript = await res.json();
        activeTranscriptSegments = transcript.segments || [];
        renderSegments(activeTranscriptSegments);
      }
    } catch (err) {
      segmentsList.innerHTML = '<div class="empty-segments">Failed to load transcript details.</div>';
    }
  }

  function renderSegments(segments) {
    segmentsList.innerHTML = '';
    if (!segments || segments.length === 0) {
      segmentsList.innerHTML = '<div class="empty-segments">No transcript text segments found.</div>';
      return;
    }

    segments.forEach(s => {
      const row = document.createElement('div');
      row.className = 'segment-row';

      const formatTime = (sec) => {
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return `${m}:${s < 10 ? '0' : ''}${s}`;
      };

      row.innerHTML = `
        <span class="seg-timestamp">${formatTime(s.start)} - ${formatTime(s.end)}</span>
        <span class="seg-text">${s.text}</span>
      `;
      segmentsList.appendChild(row);
    });
  }

  if (transcriptSearchInput) {
    transcriptSearchInput.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase();
      const filtered = activeTranscriptSegments.filter(s => s.text.toLowerCase().includes(q));
      renderSegments(filtered);
    });
  }

  // --- Quiz Generator UI Logic ---
  const generateQuizBtn = document.getElementById('generateQuizBtn');
  const quizModal = document.getElementById('quizModal');
  const closeQuizModalBtn = document.getElementById('closeQuizModalBtn');
  const quizTitle = document.getElementById('quizTitle');
  const quizProgressFill = document.getElementById('quizProgressFill');
  const quizProgressText = document.getElementById('quizProgressText');
  const quizLoading = document.getElementById('quizLoading');
  const quizQuestionContainer = document.getElementById('quizQuestionContainer');
  const quizQuestionText = document.getElementById('quizQuestionText');
  const quizOptionsGrid = document.getElementById('quizOptionsGrid');
  const quizExplanationBox = document.getElementById('quizExplanationBox');
  const expStatus = document.getElementById('expStatus');
  const expText = document.getElementById('expText');
  const quizResultContainer = document.getElementById('quizResultContainer');
  const resultScoreHeading = document.getElementById('resultScoreHeading');
  const resultScoreSub = document.getElementById('resultScoreSub');
  const nextQuizQuestionBtn = document.getElementById('nextQuizQuestionBtn');
  const retryQuizBtn = document.getElementById('retryQuizBtn');

  let currentQuizQuestions = [];
  let currentQuizIndex = 0;
  let currentQuizScore = 0;

  function openQuizModal() {
    if (quizModal) quizModal.style.display = 'flex';
    fetchQuiz();
  }

  function closeQuizModal() {
    if (quizModal) quizModal.style.display = 'none';
  }

  if (generateQuizBtn) generateQuizBtn.addEventListener('click', openQuizModal);
  if (closeQuizModalBtn) closeQuizModalBtn.addEventListener('click', closeQuizModal);
  if (retryQuizBtn) retryQuizBtn.addEventListener('click', fetchQuiz);

  async function fetchQuiz() {
    quizLoading.style.display = 'block';
    quizQuestionContainer.style.display = 'none';
    quizResultContainer.style.display = 'none';
    nextQuizQuestionBtn.style.display = 'none';
    quizProgressFill.style.width = '0%';
    quizProgressText.textContent = 'Generating questions from transcript...';

    const selectedVid = videoFilterSelect ? videoFilterSelect.value : null;

    try {
      const res = await fetchWithSession('/quiz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: selectedVid, num_questions: 5 })
      });

      if (res && res.ok) {
        const data = await res.json();
        currentQuizQuestions = data.questions || [];
        quizTitle.textContent = data.title || 'Lecture Self-Assessment';
        currentQuizIndex = 0;
        currentQuizScore = 0;
        quizLoading.style.display = 'none';

        if (currentQuizQuestions.length > 0) {
          renderQuestion(currentQuizIndex);
        } else {
          quizProgressText.textContent = 'No questions generated.';
        }
      } else {
        let errorMsg = 'Failed to load quiz.';
        try {
          const err = await res.json();
          if (err && err.detail) errorMsg = err.detail;
        } catch (_) {}
        quizLoading.innerHTML = `<p class="form-error" style="max-width: 480px; margin: 10px auto; line-height: 1.5;"><i class="fa-solid fa-circle-exclamation"></i> ${errorMsg}</p>`;
      }
    } catch (e) {
      quizLoading.innerHTML = `<p class="form-error" style="max-width: 480px; margin: 10px auto; line-height: 1.5;"><i class="fa-solid fa-circle-exclamation"></i> Could not connect to server to generate quiz. Please check server logs.</p>`;
    }
  }

  function renderQuestion(index) {
    if (index >= currentQuizQuestions.length) {
      showQuizResults();
      return;
    }

    const q = currentQuizQuestions[index];
    const total = currentQuizQuestions.length;

    quizProgressFill.style.width = `${((index + 1) / total) * 100}%`;
    quizProgressText.textContent = `Question ${index + 1} of ${total}`;

    quizQuestionText.textContent = `${index + 1}. ${q.question}`;
    quizOptionsGrid.innerHTML = '';
    quizExplanationBox.style.display = 'none';
    nextQuizQuestionBtn.style.display = 'none';
    quizQuestionContainer.style.display = 'flex';

    q.options.forEach(opt => {
      const btn = document.createElement('button');
      btn.className = 'option-btn';
      btn.innerHTML = `<i class="fa-regular fa-circle"></i> <span>${opt}</span>`;
      btn.addEventListener('click', () => handleOptionClick(btn, opt, q));
      quizOptionsGrid.appendChild(btn);
    });
  }

  function handleOptionClick(selectedBtn, selectedOption, q) {
    const allBtns = quizOptionsGrid.querySelectorAll('.option-btn');
    allBtns.forEach(b => b.disabled = true);

    const isCorrect = selectedOption.trim().toLowerCase() === q.correct_answer.trim().toLowerCase() ||
                      selectedOption.includes(q.correct_answer) ||
                      q.correct_answer.includes(selectedOption);

    if (isCorrect) {
      selectedBtn.classList.add('correct');
      selectedBtn.querySelector('i').className = 'fa-solid fa-circle-check';
      expStatus.className = 'exp-status correct';
      expStatus.textContent = '✓ Correct Answer!';
      currentQuizScore++;
    } else {
      selectedBtn.classList.add('incorrect');
      selectedBtn.querySelector('i').className = 'fa-solid fa-circle-xmark';
      expStatus.className = 'exp-status incorrect';
      expStatus.textContent = `✗ Incorrect. Correct choice: ${q.correct_answer}`;

      allBtns.forEach(b => {
        if (b.textContent.includes(q.correct_answer)) {
          b.classList.add('correct');
        }
      });
    }

    expText.textContent = `${q.explanation} (Timestamp: ${q.timestamp})`;
    quizExplanationBox.style.display = 'block';
    nextQuizQuestionBtn.style.display = 'inline-flex';
  }

  if (nextQuizQuestionBtn) {
    nextQuizQuestionBtn.addEventListener('click', () => {
      currentQuizIndex++;
      renderQuestion(currentQuizIndex);
    });
  }

  function showQuizResults() {
    quizQuestionContainer.style.display = 'none';
    nextQuizQuestionBtn.style.display = 'none';
    quizResultContainer.style.display = 'block';

    const total = currentQuizQuestions.length;
    const pct = Math.round((currentQuizScore / total) * 100);

    resultScoreHeading.textContent = `Quiz Complete!`;
    resultScoreSub.textContent = `You scored ${currentQuizScore} out of ${total} (${pct}%)`;
    quizProgressText.textContent = `Completed • Score: ${pct}%`;
  }
});
