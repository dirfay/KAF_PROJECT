(function () {
	"use strict";

	const ENDPOINT = "/ask";
	const STORAGE_KEY = "kaf_assistant_chat_v2";
	const MAX_MESSAGES = 200;

	const btn = document.getElementById("kaf-assistant-btn");
	const panel = document.getElementById("kaf-assistant-panel");
	const out = document.getElementById("kaf-chat-output");
	const input = document.getElementById("kaf-chat-input");
	const send = document.getElementById("kaf-chat-send");

	if (!btn || !panel || !out || !input || !send) {
		console.warn("Assistant UI elements not found assistant.js not initialized.");
		return;
	}

	const WELCOME_TITLE = "Вітаю! Я асистент кафедри.";
	const WELCOME_TEXT = "Я можу допомогти з питаннями про вступ, освітні програми, контакти, графік прийому, стажування, наукову діяльність та новини кафедри.";
	const EXAMPLE_QUESTIONS = [
		"Як вступити?",
		"Де знаходиться кафедра?",
		"Які є освітні програми?",
		"Який email кафедри?",
		"Чи є стажування?",
		"Які останні новини?"
	];

	function escapeHtml(str) {
		if (str == null) return "";
		return String(str)
			.replaceAll("&", "&amp;")
			.replaceAll("<", "&lt;")
			.replaceAll(">", "&gt;")
			.replaceAll('"', "&quot;")
			.replaceAll("'", "&#39;");
	}

	function textToHtmlWithBreaks(text) {
		return escapeHtml(text).replace(/\n/g, "<br/>");
	}

	function saveHistory(arr) {
		try {
			sessionStorage.setItem(STORAGE_KEY, JSON.stringify(arr.slice(-MAX_MESSAGES)));
		} catch (e) {
			console.warn("Failed to save assistant history:", e);
		}
	}

	function loadHistory() {
		try {
			const raw = sessionStorage.getItem(STORAGE_KEY);
			return raw ? JSON.parse(raw) : [];
		} catch (e) {
			return [];
		}
	}

	function scrollToBottom() {
		out.scrollTop = out.scrollHeight;
	}

	function appendMessage(msg, role, meta = {}) {
		const el = document.createElement("div");
		el.className = "kaf-msg " + (role === "user" ? "kaf-user" : "kaf-bot");
		el.setAttribute("data-role", role);

		if (role === "bot" && meta.source) {
			const s = document.createElement("div");
			s.className = "kaf-source";
			s.textContent = "Джерело: " + meta.source;
			el.innerHTML = textToHtmlWithBreaks(msg);
			el.appendChild(s);
		} else {
			el.innerHTML = textToHtmlWithBreaks(msg);
		}

		if (role === "bot") {
			const tools = document.createElement("div");
			tools.style.marginTop = "6px";
			tools.style.display = "flex";
			tools.style.gap = "8px";
			tools.style.justifyContent = "flex-end";

			const copyBtn = document.createElement("button");
			copyBtn.type = "button";
			copyBtn.className = "btn";
			copyBtn.style.padding = "4px 8px";
			copyBtn.style.fontSize = "0.85rem";
			copyBtn.textContent = "Скопіювати";
			copyBtn.addEventListener("click", async () => {
				try {
					await navigator.clipboard.writeText(msg);
					copyBtn.textContent = "Скопійовано";
					setTimeout(() => (copyBtn.textContent = "Скопіювати"), 1200);
				} catch (e) {
					copyBtn.textContent = "Помилка";
					setTimeout(() => (copyBtn.textContent = "Скопіювати"), 1200);
				}
			});

			tools.appendChild(copyBtn);
			el.appendChild(tools);
		}

		out.appendChild(el);
		scrollToBottom();
	}

	function appendWelcomeBlock() {
		const wrap = document.createElement("div");
		wrap.className = "kaf-msg kaf-bot";
		wrap.setAttribute("data-role", "bot");

		const title = document.createElement("div");
		title.style.fontWeight = "600";
		title.style.marginBottom = "6px";
		title.textContent = WELCOME_TITLE;

		const text = document.createElement("div");
		text.innerHTML = textToHtmlWithBreaks(WELCOME_TEXT);

		const hint = document.createElement("div");
		hint.style.marginTop = "10px";
		hint.style.marginBottom = "8px";
		hint.style.fontSize = "0.95rem";
		hint.style.color = "#444";
		hint.textContent = "Спробуйте одне з питань:";

		const actions = document.createElement("div");
		actions.style.display = "flex";
		actions.style.flexWrap = "wrap";
		actions.style.gap = "8px";
		actions.style.marginTop = "6px";

		EXAMPLE_QUESTIONS.forEach((question) => {
			const qBtn = document.createElement("button");
			qBtn.type = "button";
			qBtn.className = "btn";
			qBtn.style.padding = "6px 10px";
			qBtn.style.fontSize = "0.85rem";
			qBtn.textContent = question;
			qBtn.addEventListener("click", () => {
				askServer(question);
			});
			actions.appendChild(qBtn);
		});

		wrap.appendChild(title);
		wrap.appendChild(text);
		wrap.appendChild(hint);
		wrap.appendChild(actions);

		out.appendChild(wrap);
		scrollToBottom();
	}

	let history = loadHistory();
	if (history && history.length) {
		history.forEach(m => appendMessage(m.text, m.role, { source: m.source }));
	}

	function showWelcomeIfNeeded() {
		if (!history.length && !out.children.length) {
			appendWelcomeBlock();
		}
	}

	function togglePanel(show) {
		const isHidden = panel.hasAttribute("hidden");
		const shouldShow = typeof show === "boolean" ? show : isHidden;
		if (shouldShow) {
			panel.removeAttribute("hidden");
			showWelcomeIfNeeded();
			input.focus();
		} else {
			panel.setAttribute("hidden", "");
		}
	}

	btn.addEventListener("click", () => {
		const isHidden = panel.hasAttribute("hidden");
		togglePanel(isHidden);
	});

	document.addEventListener("keydown", (e) => {
		if (e.key === "Escape" && !panel.hasAttribute("hidden")) {
			togglePanel(false);
		}
	});

	async function askServer(question) {
		if (!question || !question.trim()) return;

		const q = question.trim();

		history.push({ role: "user", text: q });
		saveHistory(history);
		appendMessage(q, "user");

		appendMessage("Зачекайте, шукаю відповідь…", "bot");

		send.disabled = true;
		input.disabled = true;

		try {
			const payload = {
				question: q,
				url: window.location.href,
				title: document.title || ""
			};

			const res = await fetch(ENDPOINT, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(payload),
				credentials: "same-origin"
			});

			const botPlaceholders = Array.from(out.querySelectorAll(".kaf-msg.kaf-bot"));
			const lastPlaceholder = botPlaceholders[botPlaceholders.length - 1];
			if (lastPlaceholder && lastPlaceholder.textContent.includes("Зачекайте")) {
				lastPlaceholder.remove();
			}

			if (!res.ok) {
				let errText = `Помилка сервера ${res.status}`;
				try {
					const json = await res.json();
					errText += json.detail ? `: ${json.detail}` : "";
				} catch (e) {
					const t = await res.text();
					if (t) errText += `: ${t}`;
				}
				appendMessage(errText, "bot");
				history.push({ role: "bot", text: errText, source: "error" });
				saveHistory(history);
				return;
			}

			const json = await res.json();
			const answer = json.answer || "Відповідь не знайдена.";
			const source = json.source || "";

			appendMessage(answer, "bot", { source });
			history.push({ role: "bot", text: answer, source });
			saveHistory(history);
		} catch (err) {
			const botPlaceholders2 = Array.from(out.querySelectorAll(".kaf-msg.kaf-bot"));
			const lastPlaceholder2 = botPlaceholders2[botPlaceholders2.length - 1];
			if (lastPlaceholder2 && lastPlaceholder2.textContent.includes("Зачекайте")) {
				lastPlaceholder2.remove();
			}

			const errMsg = "Помилка запиту: " + (err && err.message ? err.message : String(err));
			appendMessage(errMsg, "bot");
			history.push({ role: "bot", text: errMsg, source: "error" });
			saveHistory(history);
		} finally {
			send.disabled = false;
			input.disabled = false;
			input.value = "";
			input.focus();
		}
	}

	send.addEventListener("click", () => {
		const q = input.value.trim();
		if (!q) return;
		askServer(q);
	});

	input.addEventListener("keydown", (e) => {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			send.click();
		}
	});

	function clearChat() {
		history = [];
		saveHistory(history);
		out.innerHTML = "";
		showWelcomeIfNeeded();
	}

	(function addClearButton() {
		const hdr = panel.querySelector("header");
		if (!hdr) return;
		const clear = document.createElement("button");
		clear.type = "button";
		clear.className = "btn";
		clear.style.fontSize = "0.85rem";
		clear.style.padding = "4px 8px";
		clear.style.marginLeft = "8px";
		clear.textContent = "Очистити";
		clear.addEventListener("click", () => {
			if (!confirm("Видалити історію чату в поточній сесії?")) return;
			clearChat();
		});
		hdr.appendChild(clear);
	})();

	if (!panel.hasAttribute("hidden")) {
		panel.setAttribute("hidden", "");
	}

	(function checkOpenParam() {
		try {
			const u = new URL(window.location.href);
			if (u.searchParams.get("assistant") === "1") {
				togglePanel(true);
			}
		} catch (e) { }
	})();

})();