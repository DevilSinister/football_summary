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

// Teams come from the database so a name can only be picked, never misspelt.
let teamListPromise = null;
const selectedTeams = {};

function loadTeams() {
  if (!teamListPromise) {
    teamListPromise = request("/api/teams").catch((error) => {
      teamListPromise = null;
      throw error;
    });
  }
  return teamListPromise;
}

function searchKey(value) {
  return String(value ?? "")
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function matchTeams(teams, query, excludeId) {
  const key = searchKey(query);
  const ranked = [];
  teams.forEach((team) => {
    if (team.team_id === excludeId) return;
    if (!key) {
      ranked.push([2, team]);
      return;
    }
    const name = searchKey(team.team_name);
    const spellings = [name, searchKey(team.short_code), ...(team.aliases || []).map(searchKey)];
    // Spaces are optional: "realmadrid" finds Real Madrid.
    const compactKey = key.replace(/ /g, "");
    const compact = spellings.map((spelling) => spelling.replace(/ /g, ""));
    if (name.startsWith(key) || compact[0].startsWith(compactKey)) ranked.push([0, team]);
    else if (spellings.some((spelling) => spelling.startsWith(key) || spelling.split(" ").some((word) => word.startsWith(key)))) ranked.push([1, team]);
    else if (compact.some((spelling) => spelling.includes(compactKey))) ranked.push([2, team]);
  });
  ranked.sort((a, b) => a[0] - b[0] || a[1].team_name.localeCompare(b[1].team_name));
  return ranked.map(([, team]) => team);
}

function renderTeamCandidates(candidates) {
  const container = document.getElementById("team-candidates");
  container.replaceChildren();
  Object.keys(selectedTeams).forEach((key) => delete selectedTeams[key]);
  candidates.forEach((candidate) => {
    const kit = candidate.cluster_id;
    const images = candidate.sample_images?.length ? candidate.sample_images : [candidate.sample_image];
    const article = document.createElement("article");
    article.className = "team-candidate";
    article.innerHTML = `
      <div class="sample-frame">
        <img alt="">
        <div class="sample-tools">
          <span class="sample-count" aria-live="polite"></span>
          <button class="sample-refresh" type="button">
            <span aria-hidden="true">↻</span> Another player
          </button>
        </div>
      </div>
      <div class="candidate-input">
        <label class="field" for="team-${kit}-search">
          <span>Team <small class="kit-label">Kit ${kit}</small></span>
        </label>
        <div class="combobox">
          <input id="team-${kit}-search" type="text" role="combobox" aria-autocomplete="list" aria-expanded="false"
            aria-controls="team-${kit}-options" autocomplete="off" autocapitalize="words" spellcheck="false"
            placeholder="Search teams, e.g. Liverpool">
          <ul id="team-${kit}-options" class="team-options" role="listbox" aria-label="Teams for kit ${kit}" hidden></ul>
        </div>
      </div>`;
    container.appendChild(article);
    setupSampleCycler(article, kit, images);
    setupTeamPicker(article.querySelector(".combobox"), kit);
  });
  loadTeams().catch((error) => setError("teams-error", `The team list could not be loaded. ${error.message}`));
}

function setupSampleCycler(article, kit, images) {
  const image = article.querySelector("img");
  const count = article.querySelector(".sample-count");
  const button = article.querySelector(".sample-refresh");
  let index = 0;
  const show = () => {
    image.src = images[index];
    image.alt = `Player wearing detected kit ${kit}, image ${index + 1} of ${images.length}`;
    count.textContent = images.length > 1 ? `${index + 1} / ${images.length}` : "";
  };
  if (images.length < 2) {
    button.hidden = true;
  } else {
    button.setAttribute("aria-label", `Show another player wearing kit ${kit}`);
    button.addEventListener("click", () => {
      index = (index + 1) % images.length;
      show();
    });
  }
  show();
}

function setupTeamPicker(box, kit) {
  const input = box.querySelector("input");
  const list = box.querySelector("[role='listbox']");
  let options = [];
  let active = -1;
  const otherKit = kit === 1 ? 2 : 1;

  const close = () => {
    list.hidden = true;
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
    active = -1;
  };
  const choose = (team) => {
    selectedTeams[kit] = team;
    input.value = team.team_name;
    box.classList.add("is-chosen");
    close();
    setError("teams-error", "");
  };
  const highlight = (index) => {
    const items = [...list.children];
    items.forEach((item, position) => item.setAttribute("aria-selected", String(position === index)));
    active = index;
    if (items[index]?.id) {
      input.setAttribute("aria-activedescendant", items[index].id);
      items[index].scrollIntoView({ block: "nearest" });
    }
  };
  const open = async () => {
    let teams;
    try {
      teams = await loadTeams();
    } catch {
      return;
    }
    options = matchTeams(teams, input.value, selectedTeams[otherKit]?.team_id).slice(0, 50);
    list.replaceChildren();
    if (!options.length) {
      const empty = document.createElement("li");
      empty.className = "team-option is-empty";
      empty.textContent = "No team in the database matches this search.";
      list.appendChild(empty);
    }
    options.forEach((team, position) => {
      const item = document.createElement("li");
      item.id = `team-${kit}-option-${team.team_id}`;
      item.className = "team-option";
      item.setAttribute("role", "option");
      item.setAttribute("aria-selected", "false");
      const name = document.createElement("span");
      name.textContent = team.team_name;
      item.appendChild(name);
      if (team.short_code) {
        const code = document.createElement("small");
        code.textContent = team.short_code;
        item.appendChild(code);
      }
      // pointerdown keeps focus in the input, so blur does not close the list first.
      item.addEventListener("pointerdown", (event) => event.preventDefault());
      item.addEventListener("click", () => choose(team));
      item.addEventListener("pointerenter", () => highlight(position));
      list.appendChild(item);
    });
    list.hidden = false;
    input.setAttribute("aria-expanded", "true");
    if (options.length && input.value.trim()) highlight(0);
  };

  input.addEventListener("focus", open);
  input.addEventListener("input", () => {
    delete selectedTeams[kit];
    box.classList.remove("is-chosen");
    open();
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (list.hidden) {
        open();
        return;
      }
      if (!options.length) return;
      const step = event.key === "ArrowDown" ? 1 : -1;
      highlight((active + step + options.length) % options.length);
    } else if (event.key === "Enter" && !list.hidden) {
      event.preventDefault();
      if (options[active]) choose(options[active]);
    } else if (event.key === "Escape") {
      close();
    }
  });
  input.addEventListener("blur", async () => {
    close();
    if (selectedTeams[kit]) {
      input.value = selectedTeams[kit].team_name;
      return;
    }
    // An exact name typed in full counts as picked.
    const key = searchKey(input.value);
    if (!key) return;
    try {
      const teams = await loadTeams();
      const exact = teams.find((team) => team.team_id !== selectedTeams[otherKit]?.team_id
        && [team.team_name, team.short_code, ...(team.aliases || [])].some((spelling) => searchKey(spelling) === key));
      if (exact) choose(exact);
    } catch {
      /* the load error is already shown */
    }
  });
}

async function confirmTeams(event) {
  event.preventDefault();
  setError("teams-error", "");
  const missing = [1, 2].find((kit) => !selectedTeams[kit]);
  if (missing) {
    setError("teams-error", `Choose the team for kit ${missing} from the list.`);
    document.getElementById(`team-${missing}-search`)?.focus();
    return;
  }
  if (selectedTeams[1].team_id === selectedTeams[2].team_id) {
    setError("teams-error", "Choose a different team for each kit.");
    return;
  }
  const button = event.currentTarget.querySelector("button[type='submit']");
  button.disabled = true;
  button.textContent = "Confirming teams…";
  try {
    await request(`/api/processing/confirm-teams/${activeJobId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        team_1_id: selectedTeams[1].team_id,
        team_2_id: selectedTeams[2].team_id,
        team_1_name: selectedTeams[1].team_name,
        team_2_name: selectedTeams[2].team_name,
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
  renderHighlights(result);
}

const EVENT_LABELS = {
  goal: "Goal",
  shot: "Shot",
  save: "Save",
  cross: "Cross",
  penalty: "Penalty",
  foul: "Foul",
  yellow_card: "Yellow card",
  red_card: "Red card",
};

function renderHighlights(result) {
  const container = document.getElementById("highlights");
  container.replaceChildren();
  const clips = (result.clips || []).filter((clip) => clip.url);
  if (!clips.length) {
    container.innerHTML = '<p class="empty-result">No highlight clips were cut for this match.</p>';
    return;
  }
  if (result.reel?.url) {
    container.appendChild(highlightCard(result.reel.url, "Match highlight reel", `${clips.length} clips joined`, true));
  }
  clips.forEach((clip) => {
    // A goal and the shot before it share one clip; name each kind once.
    const title = [...new Set(clip.event_types)].map((type) => EVENT_LABELS[type] || type).join(" · ");
    const span = `${formatTime(clip.start_seconds)}–${formatTime(clip.end_seconds)}`;
    container.appendChild(highlightCard(clip.url, title, span, false));
  });
}

function highlightCard(url, title, meta, isReel) {
  const article = document.createElement("article");
  article.className = isReel ? "highlight-clip is-reel" : "highlight-clip";
  const video = document.createElement("video");
  video.controls = true;
  video.preload = "metadata";
  video.playsInline = true;
  video.src = url;
  video.setAttribute("aria-label", title);
  const caption = document.createElement("div");
  caption.className = "highlight-caption";
  const heading = document.createElement("strong");
  heading.textContent = title;
  const detail = document.createElement("span");
  detail.textContent = meta;
  const download = document.createElement("a");
  download.href = url;
  download.download = "";
  download.textContent = "Download";
  caption.append(heading, detail, download);
  article.append(video, caption);
  return article;
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
