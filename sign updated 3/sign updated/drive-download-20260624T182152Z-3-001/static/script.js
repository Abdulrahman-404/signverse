(() => {
  const form = document.getElementById("translate-form");
  const tabs = document.querySelectorAll(".tab");
  const groups = document.querySelectorAll("[data-mode-group]");
  const submitBtn = document.getElementById("submit-btn");
  const statusBar = document.getElementById("status-bar");

  const videoPlaceholder = document.getElementById("video-placeholder");
  const avatarCanvas = document.getElementById("avatar-canvas");
  const avatarStatus = document.getElementById("avatar-status");
  const sigmlLink = document.getElementById("sigml-link");
  const videoLink = document.getElementById("video-link");

  const transcriptBlock = document.getElementById("transcript-block");
  const transcriptText = document.getElementById("transcript-text");

  const resultsDetails = document.getElementById("results-details");
  const cleanText = document.getElementById("clean-text");
  const statsText = document.getElementById("stats-text");
  const warningsBlock = document.getElementById("warnings-block");

  const breakdownBlock = document.getElementById("breakdown-block");
  const breakdown = document.getElementById("breakdown");

  const audioInput = document.getElementById("audio-input");
  const fileSelected = document.getElementById("file-selected");

  let activeMode = "text";

  // ── tab switching ──────────────────────────────────────────────────────
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      activeMode = tab.dataset.mode;
      tabs.forEach((t) => t.classList.toggle("active", t === tab));
      groups.forEach((g) => {
        g.classList.toggle("hidden", g.dataset.modeGroup !== activeMode);
      });
    });
  });

  // ── show selected file name ───────────────────────────────────────────
  audioInput.addEventListener("change", () => {
    if (audioInput.files.length) {
      const f = audioInput.files[0];
      fileSelected.textContent = `📎 ${f.name} (${(f.size / (1024 * 1024)).toFixed(1)} MB)`;
      fileSelected.classList.remove("hidden");
    } else {
      fileSelected.classList.add("hidden");
    }
  });

  // ── status bar helper (loading / error / success) ────────────────────
  function showStatus(message, type) {
    statusBar.className = `status-bar ${type}`;
    statusBar.innerHTML = message;
  }
  function hideStatus() {
    statusBar.className = "status-bar";
    statusBar.innerHTML = "";
  }

  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    if (isLoading) {
      showStatus('<span class="spinner"></span> جارِ الترجمة، قد يستغرق هذا بعض الوقت...', "loading");
    }
  }

  function renderBreakdown(groupsData) {
    breakdown.innerHTML = "";
    for (const item of groupsData) {
      if (item.matched) {
        const chip = document.createElement("span");
        chip.className = "chip-matched";
        chip.textContent = item.word;
        breakdown.appendChild(chip);
        continue;
      }

      const letters = item.letters || [];
      const fullyUnmatched = letters.length === 1 && letters[0].letter === item.word;

      if (fullyUnmatched) {
        const chip = document.createElement("span");
        chip.className = "chip-fully-unmatched";
        chip.textContent = item.word;
        breakdown.appendChild(chip);
        continue;
      }

      const group = document.createElement("div");
      group.className = "chip-unmatched-group";

      const word = document.createElement("span");
      word.className = "original-word";
      word.textContent = item.word;
      group.appendChild(word);

      const row = document.createElement("div");
      row.className = "letters-row";
      for (const l of letters) {
        const lc = document.createElement("span");
        lc.className = "letter-chip " + (l.matched ? "matched" : "unmatched");
        lc.textContent = l.letter;
        row.appendChild(lc);
      }
      group.appendChild(row);
      breakdown.appendChild(group);
    }
  }

  // ── submit ─────────────────────────────────────────────────────────────
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideStatus();

    const formData = new FormData();
    if (activeMode === "text") {
      const val = document.getElementById("text-input").value.trim();
      if (!val) return showStatus("⚠️ يرجى إدخال نص.", "error");
      formData.append("text", val);
    } else {
      if (!audioInput.files.length) return showStatus("⚠️ يرجى اختيار ملف صوتي أو مرئي.", "error");
      formData.append("audio", audioInput.files[0]);
    }

    setLoading(true);
    try {
      const resp = await fetch("/api/translate", { method: "POST", body: formData });
      const data = await resp.json();

      if (!resp.ok || data.status !== "ok") {
        showStatus("❌ " + (data.error || "حدث خطأ غير متوقع."), "error");
        return;
      }

      videoPlaceholder.style.display = "none";
      avatarCanvas.classList.remove("hidden");
      avatarStatus.classList.remove("hidden");

      try {
        const sigmlText = await (await fetch(data.sigml_url)).text();
        window.AvatarViewer && window.AvatarViewer.playSigml(sigmlText);
      } catch (err) {
        showStatus("⚠️ تعذر تحميل ملف SiGML لعرضه على المجسم.", "error");
      }

      sigmlLink.href = data.sigml_url;
      sigmlLink.classList.remove("hidden");
      videoLink.href = data.video_url;
      videoLink.classList.remove("hidden");

      if (data.transcript) {
        transcriptBlock.classList.remove("hidden");
        transcriptText.textContent = data.transcript;
      } else {
        transcriptBlock.classList.add("hidden");
      }

      resultsDetails.classList.remove("hidden");
      cleanText.textContent = data.clean_text;
      statsText.textContent =
        `عدد الوحدات: ${data.stats.total_units} | ` +
        `تطابق مباشر: ${data.stats.dictionary_hits} | ` +
        `عدد الإطارات: ${data.stats.video_frames}`;

      warningsBlock.innerHTML = "";
      for (const w of data.warnings || []) {
        const p = document.createElement("p");
        p.className = "warning-item";
        p.textContent = w;
        warningsBlock.appendChild(p);
      }

      breakdownBlock.classList.remove("hidden");
      renderBreakdown(data.breakdown || []);

      showStatus("✅ تمت الترجمة بنجاح!", "success");
    } catch (err) {
      showStatus("❌ تعذر الاتصال بالخادم.", "error");
    } finally {
      setLoading(false);
    }
  });
})();
