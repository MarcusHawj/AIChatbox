"""
app.py — Buddy AI Chat  (v2 — Modern UI)

Setup:
  pip install torch numpy customtkinter
  python train.py
  python app.py
"""

import json, random, re, time, threading, os, sys
import tkinter as tk
from tkinter import colorchooser, messagebox
import customtkinter as ctk
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def resource_path(rel):
    try:    base = sys._MEIPASS
    except: base = os.path.abspath(".")
    return os.path.join(base, rel)


# ══════════════════════════════════════════════════════════════════════════════
#  NLP  (zero external downloads)
# ══════════════════════════════════════════════════════════════════════════════

class SimpleStemmer:
    SUFFIXES = [
        ("ational","ate"),("tional","tion"),("izing","ize"),("ising","ise"),
        ("ness",""),("ment",""),("ful",""),("ous",""),("ive",""),("able",""),
        ("ible",""),("ant",""),("ent",""),("ism",""),("ate",""),("al",""),
        ("er",""),("ic",""),("ly",""),("ed",""),("ing",""),
    ]
    def stem(self, word):
        word = word.lower()
        for s in ("sses","ies"):
            if word.endswith(s): return word[:-2]
        if word.endswith(("ss","us")): return word
        if word.endswith("s") and len(word) > 3: word = word[:-1]
        for suf, rep in self.SUFFIXES:
            if word.endswith(suf) and len(word) > len(suf)+2:
                return word[:-len(suf)] + rep
        return word

_stemmer = SimpleStemmer()

def tokenize(text):  return re.findall(r"\b\w+\b", text)
def stem(word):      return _stemmer.stem(word.lower())

def bag_of_words(tokens, vocab):
    s = {stem(w) for w in tokens}
    return np.array([1.0 if w in s else 0.0 for w in vocab], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  NEURAL NETWORK
# ══════════════════════════════════════════════════════════════════════════════

class ChatNet(nn.Module):
    def __init__(self, inp, hid, out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hid),   nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hid, hid//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hid//2, out),
        )
    def forward(self, x): return self.net(x)


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERSATION AI
# ══════════════════════════════════════════════════════════════════════════════

class ConversationAI:
    THRESHOLD = 0.58

    FOLLOWUPS = {
        "why":         ["Just how it is!", "Good question — hard to say!", "It's a mystery even to me."],
        "why?":        ["Just how it is!", "Good question — hard to say!", "It's a mystery even to me."],
        "really":      ["Yep, for real!", "100%!", "As real as I can be!"],
        "really?":     ["Yep, for real!", "100%!", "I wouldn't make it up!"],
        "ok":          ["Alright!", "Cool!", "Got it!"],
        "okay":        ["Sounds good!", "Alright then!", "Nice!"],
        "lol":         ["Ha! Glad that landed 😄", "Right?!", "Comedy is my backup career."],
        "haha":        ["😄", "Glad you laughed!", "Always here for the laughs."],
        "hahaha":      ["Okay okay, I'll be here all week 😂", "That got you!"],
        "interesting": ["Right?! I thought so too.", "There's always more to it.", "Life's full of surprises."],
        "wow":         ["I know, right?!", "Pretty wild.", "Yeah, sometimes I surprise myself."],
        "nice":        ["Thanks! You're pretty nice yourself 😊", "Appreciate it!", "✨"],
        "cool":        ["Right?", "I thought so too!", "Glad you think so!"],
        "and?":        ["And... what else are you curious about?", "Keep going!", "Tell me more!"],
        "same":        ["Ha, great minds!", "We're on the same page!", "I knew it!"],
        "true":        ["Exactly!", "Glad we agree.", "Couldn't have said it better."],
    }

    def __init__(self):
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = None
        self.vocab     = []
        self.tags      = []
        self.resp_map  = {}
        self.history   = []
        self.user_name = None
        self.last_intent = None
        self.msg_count = 0

    def load(self):
        mp = resource_path("trained_model.pth")
        ip = resource_path("intents.json")
        if not os.path.exists(mp):
            return False
        d = torch.load(mp, map_location=self.device)
        self.model = ChatNet(d["input_size"], d["hidden_size"], d["output_size"]).to(self.device)
        self.model.load_state_dict(d["model_state"])
        self.model.eval()
        self.vocab = d["all_words"]
        self.tags  = d["tags"]
        with open(ip, encoding="utf-8") as f:
            data = json.load(f)
        self.resp_map = {i["tag"]: i["responses"] for i in data["intents"]}
        return True

    def _classify(self, text):
        bow = bag_of_words(tokenize(text), self.vocab)
        t   = torch.from_numpy(bow).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(t), dim=1)
        conf, idx = torch.max(probs, dim=1)
        return self.tags[idx.item()], conf.item()

    def _extract_name(self, text):
        m = re.search(r"(?:i'?m|i am|call me|my name'?s?|name is)\s+([A-Za-z]+)", text, re.I)
        if m:
            n = m.group(1).capitalize()
            if n.lower() not in {"good","fine","okay","great","here","back",
                                  "just","not","doing","happy","sad","tired"}:
                return n
        return None

    def reply(self, user_text: str) -> str:
        self.msg_count += 1
        cleaned = user_text.strip().lower()

        name = self._extract_name(user_text)
        if name:
            self.user_name = name

        if cleaned in self.FOLLOWUPS:
            resp = random.choice(self.FOLLOWUPS[cleaned])
            self._record("user", user_text, "followup")
            self._record("ai", resp, "followup")
            return resp

        intent, conf = self._classify(user_text)

        if conf >= self.THRESHOLD:
            resp = random.choice(self.resp_map.get(intent, ["Hmm, not sure about that one."]))
        else:
            resp = random.choice([
                "Hmm, I'm not quite sure what you mean — can you rephrase?",
                "I didn't catch that. Try saying it differently?",
                "Still learning! Could you put it another way?",
                "My brain glitched a bit there 😅 What did you mean?",
            ])
            intent = "unknown"

        if self.user_name and random.random() < 0.15:
            resp = random.choice([f"Hey {self.user_name}, ", f"{self.user_name} — "]) \
                   + resp[0].lower() + resp[1:]

        if self.msg_count == 10 and intent != "goodbye":
            resp += " (Really enjoying this chat, by the way!)"

        self._record("user", user_text, intent)
        self._record("ai", resp, intent)
        self.last_intent = intent
        return resp

    def _record(self, role, text, intent):
        self.history.append({"role": role, "text": text, "intent": intent})
        if len(self.history) > 60:
            self.history = self.history[-60:]


# ══════════════════════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════════════════════

DEFAULTS = {
    "window_bg":   "#0f0f1a",
    "chat_bg":     "#13131f",
    "user_bubble": "#4f46e5",
    "ai_bubble":   "#1c1c2e",
    "input_bg":    "#1c1c2e",
    "top_bar":     "#09090f",
    "accent":      "#818cf8",
    "text":        "#e2e8f0",
    "subtext":     "#64748b",
}

COLOR_LABELS = [
    ("window_bg",   "Window Background"),
    ("chat_bg",     "Chat Background"),
    ("user_bubble", "Your Message Bubbles"),
    ("ai_bubble",   "AI Message Bubbles"),
    ("input_bg",    "Input Box"),
    ("top_bar",     "Top & Bottom Bars"),
    ("accent",      "Accent / Buttons"),
    ("text",        "Message Text"),
    ("subtext",     "Timestamps & Labels"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  SETTINGS WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.app     = parent
        self.pending = dict(parent.colors)
        self.swatches = {}

        self.title("Customize Buddy")
        self.geometry("420x580")
        self.minsize(420, 480)
        self.resizable(True, True)
        self.configure(fg_color=self.app.colors["window_bg"])
        self.grab_set()

        self._build()

    def _build(self):
        C = self.app.colors

        # Header — always visible at top
        header = ctk.CTkFrame(self, fg_color=C["top_bar"], corner_radius=0, height=56)
        header.pack(fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(header, text="✨  Customize",
                     font=ctk.CTkFont(size=17, weight="bold"),
                     text_color=C["text"]).pack(side="left", padx=18, pady=14)

        # Apply/Reset bar — pinned to bottom, always visible
        btn_bar = ctk.CTkFrame(self, fg_color=C["top_bar"], corner_radius=0, height=66)
        btn_bar.pack(fill="x", side="bottom")
        btn_bar.pack_propagate(False)

        ctk.CTkButton(
            btn_bar, text="↺  Reset",
            width=140, height=42, corner_radius=21,
            fg_color=C["ai_bubble"], hover_color="#2d2d45",
            font=ctk.CTkFont(size=13), text_color=C["text"],
            command=self._reset
        ).pack(side="left", padx=16, pady=12)

        ctk.CTkButton(
            btn_bar, text="✓  Apply",
            width=140, height=42, corner_radius=21,
            fg_color=C["accent"], hover_color="#6366f1",
            font=ctk.CTkFont(size=13, weight="bold"), text_color="#ffffff",
            command=self._apply
        ).pack(side="right", padx=16, pady=12)

        # Scrollable middle content
        scroll = ctk.CTkScrollableFrame(self, fg_color=C["window_bg"], corner_radius=0)
        scroll.pack(fill="both", expand=True)

        # Bot name card
        self._section_label(scroll, "BOT NAME")
        name_card = ctk.CTkFrame(scroll, fg_color=C["ai_bubble"], corner_radius=14)
        name_card.pack(fill="x", padx=16, pady=(0, 16))
        self.name_var = tk.StringVar(value=self.app.bot_name)
        ctk.CTkEntry(
            name_card, textvariable=self.name_var,
            font=ctk.CTkFont(size=13), height=44, corner_radius=10,
            border_width=0, fg_color="transparent", text_color=C["text"]
        ).pack(fill="x", padx=10, pady=8)

        # Colors card
        self._section_label(scroll, "COLORS")
        colors_card = ctk.CTkFrame(scroll, fg_color=C["ai_bubble"], corner_radius=14)
        colors_card.pack(fill="x", padx=16, pady=(0, 16))

        for i, (key, label) in enumerate(COLOR_LABELS):
            row = ctk.CTkFrame(colors_card, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=8)

            ctk.CTkLabel(row, text=label,
                         font=ctk.CTkFont(size=12), text_color=C["text"],
                         anchor="w").pack(side="left", fill="x", expand=True)

            swatch = tk.Frame(row, width=28, height=28,
                              bg=self.pending[key], relief="flat", cursor="hand2")
            swatch.pack(side="left", padx=(0, 8))
            swatch.pack_propagate(False)
            swatch.bind("<Button-1>", lambda e, k=key: self._pick(k))
            self.swatches[key] = swatch

            ctk.CTkButton(
                row, text="Pick", width=58, height=30,
                corner_radius=8, font=ctk.CTkFont(size=11),
                fg_color=C["accent"], hover_color="#6366f1",
                command=lambda k=key: self._pick(k)
            ).pack(side="left")

            if i < len(COLOR_LABELS) - 1:
                tk.Frame(colors_card, bg="#ffffff10", height=1).pack(fill="x", padx=14)

    def _section_label(self, parent, text):
        ctk.CTkLabel(parent, text=text,
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=self.app.colors["subtext"]).pack(
            anchor="w", padx=20, pady=(14, 6))

    def _pick(self, key):
        result = colorchooser.askcolor(color=self.pending[key], title=f"Choose — {key}")
        if result and result[1]:
            self.pending[key] = result[1]
            self.swatches[key].configure(bg=result[1])

    def _reset(self):
        self.pending = dict(DEFAULTS)
        for key, swatch in self.swatches.items():
            swatch.configure(bg=DEFAULTS[key])

    def _apply(self):
        new_name = self.name_var.get().strip() or self.app.bot_name
        self.app.apply_theme(self.pending, new_name)
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CHAT APP
# ══════════════════════════════════════════════════════════════════════════════

class ChatApp(ctk.CTk):

    def __init__(self, ai: ConversationAI):
        super().__init__()
        self.ai       = ai
        self.colors   = dict(DEFAULTS)
        self.bot_name = "Buddy"

        self.title(f"{self.bot_name} AI")
        self.geometry("940x720")
        self.minsize(620, 500)
        self.configure(fg_color=self.colors["window_bg"])

        self._typing_widget = None
        self._build()

        self.after(600, lambda: self._add_bubble(
            f"Hey! I'm {self.bot_name} 👋  Talk to me like you'd talk to a friend.",
            sender="ai"
        ))

    def _build(self):
        C = self.colors

        # Top bar
        self.topbar = ctk.CTkFrame(self, fg_color=C["top_bar"],
                                   height=60, corner_radius=0)
        self.topbar.pack(fill="x", side="top")
        self.topbar.pack_propagate(False)

        self.title_lbl = ctk.CTkLabel(
            self.topbar, text=f"  💬  {self.bot_name}",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color=C["text"]
        )
        self.title_lbl.pack(side="left", padx=(8, 4))

        ctk.CTkLabel(self.topbar, text="● Online",
                     font=ctk.CTkFont(size=11),
                     text_color="#4ade80").pack(side="left")

        self.settings_btn = ctk.CTkButton(
            self.topbar, text="⚙  Customize",
            width=124, height=36, corner_radius=18,
            font=ctk.CTkFont(size=12),
            fg_color=C["accent"], hover_color="#6366f1",
            command=lambda: SettingsWindow(self)
        )
        self.settings_btn.pack(side="right", padx=16, pady=12)

        # Bottom bar
        self.bottombar = ctk.CTkFrame(self, fg_color=C["top_bar"],
                                      height=70, corner_radius=0)
        self.bottombar.pack(fill="x", side="bottom")
        self.bottombar.pack_propagate(False)

        self.input_box = ctk.CTkEntry(
            self.bottombar,
            placeholder_text=f"  Message {self.bot_name}...",
            font=ctk.CTkFont(size=14), height=46,
            corner_radius=23, border_width=0,
            fg_color=C["input_bg"], text_color=C["text"],
            placeholder_text_color=C["subtext"]
        )
        self.input_box.pack(side="left", fill="x", expand=True,
                            padx=(16, 8), pady=12)
        self.input_box.bind("<Return>", lambda e: self._send())

        self.send_btn = ctk.CTkButton(
            self.bottombar, text="↑",
            width=46, height=46, corner_radius=23,
            font=ctk.CTkFont(size=20, weight="bold"),
            fg_color=C["accent"], hover_color="#6366f1",
            command=self._send
        )
        self.send_btn.pack(side="right", padx=(0, 16), pady=12)

        # Scrollable chat area
        self.chat_scroll = ctk.CTkScrollableFrame(
            self, fg_color=C["chat_bg"], corner_radius=0,
            scrollbar_button_color=C["subtext"],
            scrollbar_button_hover_color=C["accent"]
        )
        self.chat_scroll.pack(fill="both", expand=True)
        tk.Frame(self.chat_scroll, bg=C["chat_bg"], height=10).pack()

        self.input_box.focus_set()

    def _add_bubble(self, text: str, sender: str = "ai"):
        C       = self.colors
        is_user = sender == "user"
        color   = C["user_bubble"] if is_user else C["ai_bubble"]
        anchor  = "e" if is_user else "w"
        pad_l   = (120, 12) if is_user else (12, 120)
        label   = "You" if is_user else self.bot_name

        row = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        row.pack(fill="x", pady=4, padx=8)

        bubble = ctk.CTkFrame(row, fg_color=color, corner_radius=20)
        bubble.pack(anchor=anchor, padx=pad_l)

        ctk.CTkLabel(bubble, text=label,
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=C["subtext"]).pack(anchor="w", padx=16, pady=(12, 2))

        ctk.CTkLabel(bubble, text=text,
                     font=ctk.CTkFont(size=13),
                     text_color=C["text"],
                     wraplength=500, justify="left", anchor="w"
                     ).pack(anchor="w", padx=16, pady=(0, 4))

        ts = time.strftime("%I:%M %p").lstrip("0")
        ctk.CTkLabel(bubble, text=ts,
                     font=ctk.CTkFont(size=9),
                     text_color=C["subtext"]).pack(anchor="e", padx=16, pady=(0, 10))

        self.after(60, self._scroll_bottom)

    def _show_typing(self):
        C = self.colors
        row = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        row.pack(fill="x", pady=4, padx=8)
        inner = ctk.CTkFrame(row, fg_color=C["ai_bubble"], corner_radius=20)
        inner.pack(anchor="w", padx=(12, 120))
        ctk.CTkLabel(inner, text=f"  {self.bot_name} is typing...",
                     font=ctk.CTkFont(size=12, slant="italic"),
                     text_color=C["subtext"]).pack(padx=14, pady=14)
        self._typing_widget = row
        self.after(60, self._scroll_bottom)

    def _hide_typing(self):
        if self._typing_widget:
            self._typing_widget.destroy()
            self._typing_widget = None

    def _scroll_bottom(self):
        try:
            self.chat_scroll._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    def _send(self):
        text = self.input_box.get().strip()
        if not text:
            return
        self.input_box.delete(0, "end")
        self._add_bubble(text, sender="user")
        self.send_btn.configure(state="disabled")
        self._show_typing()

        def respond():
            time.sleep(random.uniform(0.7, 1.5))
            resp = self.ai.reply(text)
            self.after(0, lambda: self._finish(resp))

        threading.Thread(target=respond, daemon=True).start()

    def _finish(self, resp: str):
        self._hide_typing()
        self._add_bubble(resp, sender="ai")
        self.send_btn.configure(state="normal")
        self.input_box.focus_set()

    def apply_theme(self, new_colors: dict, new_name: str = None):
        self.colors.update(new_colors)
        C = self.colors

        if new_name and new_name != self.bot_name:
            self.bot_name = new_name
            self.title(f"{self.bot_name} AI")
            self.title_lbl.configure(text=f"  💬  {self.bot_name}")
            self.input_box.configure(placeholder_text=f"  Message {self.bot_name}...")

        self.configure(fg_color=C["window_bg"])
        self.topbar.configure(fg_color=C["top_bar"])
        self.bottombar.configure(fg_color=C["top_bar"])
        self.chat_scroll.configure(fg_color=C["chat_bg"])
        self.input_box.configure(fg_color=C["input_bg"], text_color=C["text"])
        self.send_btn.configure(fg_color=C["accent"])
        self.settings_btn.configure(fg_color=C["accent"])

        self._add_bubble("Theme updated! Looking fresh. 🎨", sender="ai")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ai = ConversationAI()
    if not ai.load():
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Model Not Found",
            "No trained model found!\n\n"
            "Please run train.py first:\n"
            "    python train.py\n\n"
            "Then launch the app again."
        )
        root.destroy()
        return

    app = ChatApp(ai)
    app.mainloop()


if __name__ == "__main__":
    main()
