const views = ["upload", "progress", "teams", "results"];
let activeJobId = null;
let pollTimer = null;

function showView(name) {
  views.forEach((view) => {
    document.getElementById(`${view}-view`).hidden = view !== name;
  });
  const stage = name === "progress" ? (document.body.dataset.stage || "upload") : name;
  document.querySelectorAll(".step-rail li").forEach((item) => {
    const order = ["upload", "teams", "results"];
    const itemIndex = order.indexOf(item.dataset.step);
    const stageIndex = order.indexOf(stage);
    item.classList.toggle("is-active", itemIndex === stageIndex);
    item.classList.toggle("is-complete", itemIndex < stageIndex);
  });
}

function setError(id, message) {
  const element = document.getElementById(id);
  element.textContent = message || "";
  element.hidden = !message;
}

function humanError(error) {
  if (typeof error === "string") return error;
  if (Array.isArray(error)) return error.map((item) => item.msg).join(" ");
  return error?.detail || error?.message || "Something went wrong. Please try again.";
}

async function request(url, options = {}) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(humanError(payload));
  return payload;
}

async function submitVideo(event, source) {
  event.preventDefault();
  setError("upload-error", "");
  const button = event.currentTarget.querySelector("button[type='submit']");
  button.disabled = true;
  button.textContent = "Uploading video…";
  try {
    let payload;
    if (source === "file") {
      const file = document.getElementById("video-file").files[0];
      if (!file) throw new Error("Choose a match video before continuing.");
      const form = new FormData();
      form.append("video", file);
      payload = await request("/api/processing/submit-file", { method: "POST", body: form });
    } else {
      const videoUrl = document.getElementById("video-url").value.trim();
      if (!videoUrl) throw new Error("Enter a direct video URL before continuing.");
      payload = await request("/api/processing/submit-url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_url: videoUrl }),
      });
    }
    activeJobId = payload.job_id;
    document.body.dataset.stage = "upload";
    showView("progress");
    pollStatus();
  } catch (error) {
    setError("upload-error", error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = 'Detect the teams <span aria-hidden="true">→</span>';
  }
}

async function pollStatus() {
  clearTimeout(pollTimer);
  if (!activeJobId) return;
  try {
    const job = await request(`/api/processing/status/${activeJobId}`);
    document.getElementById("progress-title").textContent = job.stage || "Analysing match";
    document.getElementById("progress-percent").textContent = `${job.progress_percent || 0}%`;
    document.getElementById("progress-bar").style.transform = `scaleX(${(job.progress_percent || 0) / 100})`;
    if (job.status === "awaiting_team_confirmation") {
      renderTeamCandidates(job.team_candidates || []);
      showView("teams");
      return;
    }
    if (job.status === "completed") {
      const payload = await request(`/api/processing/result/${activeJobId}`);
      renderResults(payload.result);
      showView("results");
      return;
    }
    if (job.status === "failed") throw new Error(job.error || "Video processing failed.");
    document.body.dataset.stage = job.status === "processing" ? "teams" : "upload";
    pollTimer = setTimeout(pollStatus, 1800);
  } catch (error) {
    document.getElementById("progress-title").textContent = "Analysis stopped";
    document.getElementById("progress-copy").textContent = `${error.message} Return to the upload step and try another video.`;
    document.getElementById("progress-bar").style.transform = "scaleX(0)";
  }
}

function renderTeamCandidates(candidates) {
  const container = document.getElementById("team-candidates");
  container.replaceChildren();
  candidates.forEach((candidate) => {
    const article = document.createElement("article");
    article.className = "team-candidate";
    article.innerHTML = `
      <img src="${candidate.sample_image}" alt="Representative player for detected team ${candidate.cluster_id}">
      <div class="candidate-input">
        <label class="field">
          <span>Team name <small class="kit-label">Kit ${candidate.cluster_id}</small></span>
          <input name="team_${candidate.cluster_id}_name" required maxlength="100" autocomplete="organization" placeholder="e.g. Liverpool">
        </label>
      </div>`;
    container.appendChild(article);
  });
}

async function confirmTeams(event) {
  event.preventDefault();
  setError("teams-error", "");
  const data = new FormData(event.currentTarget);
  const button = event.currentTarget.querySelector("button[type='submit']");
  button.disabled = true;
  button.textContent = "Confirming teams…";
  try {
    await request(`/api/processing/confirm-teams/${activeJobId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        team_1_name: data.get("team_1_name"),
        team_2_name: data.get("team_2_name"),
      }),
    });
    document.body.dataset.stage = "teams";
    document.getElementById("progress-copy").textContent = "The team names are locked in. We’re detecting events, reading shirts, and building the score sheet.";
    showView("progress");
    pollStatus();
  } catch (error) {
    setError("teams-error", error.message);
  } finally {
    button.disabled = false;
    button.innerHTML = 'Confirm teams &amp; finish analysis <span aria-hidden="true">→</span>';
  }
}

function renderResults(result) {
  const score = result.score_sheet || [];
  const left = score[0] || { team_name: "Team 1", score: 0 };
  const right = score[1] || { team_name: "Team 2", score: 0 };
  document.getElementById("score-sheet").innerHTML = `
    <div class="score-team"><strong>${escapeHtml(left.team_name)}</strong><span>Detected kit 1</span></div>
    <div class="score-number"><b>${left.score}</b><i></i><b>${right.score}</b></div>
    <div class="score-team"><strong>${escapeHtml(right.team_name)}</strong><span>Detected kit 2</span></div>`;

  const players = document.getElementById("player-results");
  players.replaceChildren();
  if (!result.players?.length) {
    players.innerHTML = '<p class="empty-result">No confirmed scorer identities were found. The final score remains available above, and uncertain shirt details have not been guessed.</p>';
  } else {
    result.players.forEach((player) => {
      const row = document.createElement("article");
      row.className = "player-result";
      const image = player.player_image
        ? `<img src="${player.player_image}" alt="Detected scorer for ${escapeHtml(player.team_name)}">`
        : '<span class="player-placeholder" aria-hidden="true">•</span>';
      const name = player.player_name || "Name not confirmed";
      const number = player.jersey_number == null ? "—" : `#${player.jersey_number}`;
      row.innerHTML = `${image}<div><strong>${escapeHtml(name)}</strong><span>${escapeHtml(player.team_name)} · ${formatTime(player.time_seconds)}</span></div><div class="player-number">${number}</div>`;
      players.appendChild(row);
    });
  }
  document.getElementById("annotated-video").href = result.annotated_video;
}

function escapeHtml(value) {
  const node = document.createElement("span");
  node.textContent = String(value ?? "");
  return node.innerHTML;
}

function formatTime(seconds) {
  const total = Math.max(0, Math.round(seconds || 0));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function reset() {
  activeJobId = null;
  clearTimeout(pollTimer);
  document.body.dataset.stage = "upload";
  document.getElementById("file-panel").reset();
  document.getElementById("url-panel").reset();
  document.getElementById("file-note").textContent = "or drop it in this area";
  showView("upload");
}

document.getElementById("file-panel").addEventListener("submit", (event) => submitVideo(event, "file"));
document.getElementById("url-panel").addEventListener("submit", (event) => submitVideo(event, "url"));
document.getElementById("teams-form").addEventListener("submit", confirmTeams);
document.getElementById("new-analysis").addEventListener("click", reset);
document.getElementById("video-file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  document.getElementById("file-note").textContent = file ? `${file.name} · ${(file.size / 1048576).toFixed(1)} MB` : "or drop it anywhere in this area";
});

document.querySelectorAll(".source-tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    const fileSelected = tab.id === "file-tab";
    document.getElementById("file-panel").hidden = !fileSelected;
    document.getElementById("url-panel").hidden = fileSelected;
    document.querySelectorAll(".source-tab").forEach((item) => {
      const selected = item === tab;
      item.classList.toggle("is-selected", selected);
      item.setAttribute("aria-selected", String(selected));
    });
  });
  tab.addEventListener("keydown", (event) => {
    const tabs = [...document.querySelectorAll(".source-tab")];
    const current = tabs.indexOf(tab);
    const target = event.key === "ArrowRight"
      ? tabs[(current + 1) % tabs.length]
      : event.key === "ArrowLeft"
        ? tabs[(current - 1 + tabs.length) % tabs.length]
        : event.key === "Home"
          ? tabs[0]
          : event.key === "End"
            ? tabs[tabs.length - 1]
            : null;
    if (target) {
      event.preventDefault();
      target.focus();
      target.click();
    }
  });
});

const dropzone = document.querySelector(".dropzone");
const videoInput = document.getElementById("video-file");
["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
  });
});
dropzone.addEventListener("drop", (event) => {
  const allowedExtensions = [".mp4", ".mov", ".avi", ".mkv"];
  const file = [...event.dataTransfer.files].find((candidate) =>
    allowedExtensions.some((extension) => candidate.name.toLowerCase().endsWith(extension))
  );
  if (!file) {
    setError("upload-error", "Drop a supported video file: MP4, MOV, AVI or MKV.");
    return;
  }
  const transfer = new DataTransfer();
  transfer.items.add(file);
  videoInput.files = transfer.files;
  videoInput.dispatchEvent(new Event("change", { bubbles: true }));
});

showView("upload");
