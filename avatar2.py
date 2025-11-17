"""
Hovoriaci avatar – OFFLINE (pygame + eSpeak‑NG)
------------------------------------------------
Stabilná verzia s:
• dvojpanelovým UI (vľavo avatar, vpravo ovládanie),
• lip‑syncom cez RMS + vyhladzovanie a 5 tvarov úst,
• AUTO emóciami z textu (SK/EN),
• „emočné okná“,
• plne funkčným textovým poľom (Ctrl+A/C/V/X, výber myšou).
+ Tmavý/svetlý režim (prepínač, ukladanie do config.json)
+ Idle animácia (blikanie očí každých 3-6 sekúnd)
+ Rolovanie textu počas reči (aktuálne slovo modré)

Požiadavky: pip install pygame numpy pyperclip + eSpeak-NG
"""

import os, subprocess, tempfile, wave, time, re, json, random
from typing import List, Tuple
import numpy as np
import pygame
import pyperclip  # <-- pre Ctrl+C/V/X

# ----------------------------- Konštanty -----------------------------
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')
CONFIG_PATH = 'config.json'  # pre ukladanie témy

#WINDOW_W = 480
#WINDOW_H = 740
WINDOW_W = 960          # 2× širšie okno
WINDOW_H = 620          # nižšie, UI vpravo



# Predvolené farby (tmavý režim)
BG = (16, 18, 22);
PANEL = (26, 28, 34);
CARD = (32, 35, 42);
TXT = (230, 230, 235);
MUTED = (165, 170, 180);
ACC = (120, 180, 255)
FONT_NAME = 'freesansbold.ttf'

# Lip-sync
FRAME_MS = 20
RMS_THR1 = 0.018
RMS_THR2 = 0.030
RMS_THR3 = 0.045
RMS_THR4 = 0.070
RMS_SMOOTH = 0.6
RMS_METER_CLAMP = 0.12

# TTS
ESPEAK_VOICE_SK = 'sk+f3'
ESPEAK_VOICE_EN = 'en+m7'
ESPEAK_WPM = 120

# Layout
LEFT_W = 480            # ľavý panel = avatar
RIGHT_W = 480           # pravý panel = UI
PAD = 16;
GAP = 10;
CTRL_H = 44;
SMALL_H = 18

# Layout – pridaj toto niekde nad App triedu (napr. pod WINDOW_H)
LEFT_X  = PAD
RIGHT_X = LEFT_W + PAD

# ----------------------------- Načítanie a ukladanie configu pre tému -----------------------------
def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {"theme": "dark"}  # default tmavý


def save_config(config):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f)


CONFIG = load_config()


# ----------------------------- Aplikácia témy -----------------------------
def apply_theme(theme):
    global BG, PANEL, CARD, TXT, MUTED, ACC
    if theme == "dark":
        BG = (16, 18, 22);
        PANEL = (26, 28, 34);
        CARD = (32, 35, 42)
        TXT = (230, 230, 235);
        MUTED = (165, 170, 180);
        ACC = (120, 180, 255)
    else:  # light
        BG = (240, 240, 245);
        PANEL = (220, 220, 230);
        CARD = (255, 255, 255)
        TXT = (30, 30, 35);
        MUTED = (100, 100, 110);
        ACC = (0, 120, 255)


# Načítať tému pri štarte
apply_theme(CONFIG["theme"])


# ----------------------------- Utility -----------------------------
def load_img(name: str) -> pygame.Surface:
    p = os.path.join(ASSETS_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f'Chýba súbor: {p}')
    return pygame.image.load(p).convert_alpha()


def espeak_wav(text: str, lang: str, wpm: int) -> str:
    voice = ESPEAK_VOICE_SK if lang == 'SK' else ESPEAK_VOICE_EN
    fd, wav_path = tempfile.mkstemp(suffix='.wav');
    os.close(fd)
    cmd = ['espeak-ng', '-v', voice, '-s', str(wpm), '-w', wav_path, text]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError('espeak-ng nie je v PATH. Doinštaluj eSpeak NG.')
    return wav_path


def wav_rms_timeline(path: str, frame_ms: int = FRAME_MS) -> Tuple[List[float], int, float]:
    with wave.open(path, 'rb') as wf:
        ch = wf.getnchannels();
        sw = wf.getsampwidth();
        sr = wf.getframerate();
        n = wf.getnframes()
        raw = wf.readframes(n)
    if sw == 2:
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError('Nepodporovaný WAV')
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    frame_len = int(sr * frame_ms / 1000)
    rms = [float(np.sqrt(np.mean(np.square(a[i:i + frame_len])))) for i in range(0, len(a), frame_len)]
    duration = len(a) / sr
    return rms, sr, duration


# ----------------------------- Text → Emócia -----------------------------
POS_WORDS = [
    'teším sa', 'tesim sa', 'veľmi rád', 'velmi rad', 'super', 'skvel', 'paráda', 'parada', 'radosť',
    'mam rad', 'mám rád', 'milujem', 'great', 'awesome', 'happy', 'glad', 'love'
]
SURP_WORDS = ['čože', 'coze', 'wow', 'neverím', 'neverim', 'vážne', 'vazne', 'fíha', 'fiha', 'prekvap', '?!', '!?']

_word_re = re.compile(r"[\wáäčďéíľĺňóôŕřšťúýž]+", re.IGNORECASE)


# 1. Emočné okná – presne na slovo + krátky "dozvuk"
def find_emotion_windows(text: str) -> List[Tuple[float, float, str]]:
    t = text.lower()
    tokens = _word_re.findall(t)
    if not tokens:
        return []

    total_words = len(tokens)
    windows = []

    for i, word in enumerate(tokens):
        word_clean = word.replace(',', '').replace('.', '').replace('!', '').replace('?', '')
        position = i / total_words                     # kde v reči je slovo (0.0 – 1.0)

        emo = None
        duration = 0.22  # koľko sekúnd trvá emócia po slove (uprav podľa potreby)

        # Surprised má prednosť
        if any(k in word_clean for k in ["wow", "čože", "coze", "fíha", "fiha", "vážne", "vazne", "neverím", "neverim", "prekvap", "!?"]):
            emo = "Surprised"
            duration = 0.35  # prekvapenie trvá dlhšie (viac dramatické)

        elif any(k in word_clean for k in ["teším sa", "tesim sa", "super", "skvelé", "paráda", "parada", "milujem", "rád", "rad", "jéé", "jupí", "hurá"]):
            emo = "Happy"

        if emo:
            start = position
            end   = min(1.0, position + duration)
            windows.append((start, end, emo))

    return windows


def base_emotion(text: str) -> str:
    t = text.lower()
    if any(k.replace(' ', '') in t.replace(' ', '') for k in POS_WORDS):
        return 'Happy'
    if any(k.replace(' ', '') in t.replace(' ', '') for k in SURP_WORDS):
        return 'Surprised'
    return 'Neutral'


# ----------------------------- UI prvky -----------------------------
class Button:
    def __init__(self, rect, label, on_click):
        self.rect = pygame.Rect(rect);
        self.label = label;
        self.on_click = on_click

    def draw(self, surf, font):
        pygame.draw.rect(surf, CARD, self.rect, border_radius=8)
        pygame.draw.rect(surf, ACC, self.rect, width=2, border_radius=8)
        txt = font.render(self.label, True, TXT)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            self.on_click()


class Dropdown:
    def __init__(self, rect, options, selected_idx=0, label=''):
        self.rect = pygame.Rect(rect)
        self.options = options
        self.selected_idx = selected_idx
        self.label = label
        self.open = False
        self.dropdown_rect = pygame.Rect(
            rect.x, rect.y + CTRL_H + SMALL_H,
            rect.width, len(options) * CTRL_H
        )

    @property
    def value(self):
        return self.options[self.selected_idx]

    def toggle(self):
        self.open = not self.open

    def select(self, idx):
        self.selected_idx = idx
        self.open = False

    def draw(self, surf, font, font_small):
        # Label
        lab = font_small.render(self.label, True, MUTED)
        surf.blit(lab, (self.rect.x + 2, self.rect.y))

        # Hlavný box
        main_box = pygame.Rect(self.rect.x, self.rect.y + SMALL_H, self.rect.width, CTRL_H)
        pygame.draw.rect(surf, CARD, main_box, border_radius=8)
        pygame.draw.rect(surf, ACC if self.open else (70, 70, 80), main_box, width=2, border_radius=8)
        surf.blit(font.render(self.value, True, TXT), (main_box.x + 12, main_box.y + 10))

        # Šípka dole
        arrow_x = main_box.right - 30
        arrow_y = main_box.centery
        points = [(arrow_x, arrow_y - 5), (arrow_x + 10, arrow_y - 5), (arrow_x + 5, arrow_y + 3)]
        pygame.draw.polygon(surf, TXT if self.open else MUTED, points)

        # Otvorená ponuka
        if self.open:
            drop_h = len(self.options) * CTRL_H
            drop_y = main_box.bottom + 2
            screen_h = surf.get_height()
            if drop_y + drop_h > screen_h:
                drop_y = main_box.y - drop_h - 2
            self.dropdown_rect = pygame.Rect(main_box.x, drop_y, main_box.width, drop_h)
            pygame.draw.rect(surf, CARD, self.dropdown_rect, border_radius=8)
            pygame.draw.rect(surf, (70, 70, 80), self.dropdown_rect, width=2, border_radius=8)
            for i, opt in enumerate(self.options):
                item_rect = pygame.Rect(
                    self.dropdown_rect.x, self.dropdown_rect.y + i * CTRL_H,
                    self.dropdown_rect.width, CTRL_H
                )
                if i == self.selected_idx:
                    pygame.draw.rect(surf, (50, 50, 60), item_rect, border_radius=8)
                txt = font.render(opt, True, TXT if i != self.selected_idx else ACC)
                surf.blit(txt, (item_rect.x + 12, item_rect.y + 10))

    def handle(self, event):
        main_box = pygame.Rect(self.rect.x, self.rect.y + SMALL_H, self.rect.width, CTRL_H)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if main_box.collidepoint(event.pos):
                self.toggle()
                return
            if self.open and self.dropdown_rect.collidepoint(event.pos):
                rel_y = event.pos[1] - self.dropdown_rect.y
                idx = rel_y // CTRL_H
                if 0 <= idx < len(self.options):
                    self.select(idx)
                return
            if self.open:
                self.open = False


# ----------------------------- TextBox s výberom -----------------------------
# ----------------------------- TextBox s výberom -----------------------------
# ============================== MULTILINE TEXTBOX (ako Word) ==============================
# ============================== MULTILINE TEXTBOX – FINÁLNA VERZIA (ako Word) ==============================
class MultiLineTextBox:
    def __init__(self, rect, font, text='', placeholder='Napíš text…'):
        self.rect = pygame.Rect(rect)
        self.font = font
        self.text = text
        self.placeholder = placeholder
        self.active = False
        self.caret = len(text)
        self.select_start = None
        self.scroll_y = 0
        self.blink_timer = 0
        self.dragging_scroll = False

    def _get_lines(self):
        max_w = self.rect.width - 40
        words = self.text.split(' ')
        lines = []
        cur = []
        for word in words:
            test = (' '.join(cur) + ' ' + word).strip() if cur else word
            if self.font.size(test)[0] <= max_w:
                cur.append(word)
            else:
                if cur:
                    lines.append(' '.join(cur))
                    cur = [word]
                else:
                    # rozbijeme extrémne dlhé slovo
                    while word:
                        for i in range(len(word), 0, -1):
                            if self.font.size(word[:i])[0] <= max_w:
                                lines.append(word[:i])
                                word = word[i:]
                                break
                        else:
                            lines.append(word)
                            word = ''
                    cur = []
        if cur:
            lines.append(' '.join(cur))
        return lines

    def _char_to_pos(self, char_idx):
        lines = self._get_lines()
        pos = 0
        for row, line in enumerate(lines):
            if pos + len(line) >= char_idx:
                return row, char_idx - pos
            pos += len(line) + 1
        return len(lines) - 1, len(lines[-1]) if lines else 0

    def _pos_to_char(self, row, col):
        lines = self._get_lines()
        pos = 0
        for r, line in enumerate(lines):
            if r == row:
                return min(pos + col, pos + len(line))
            pos += len(line) + 1
        return len(self.text)

    def handle(self, event):
        # Aktivácia po kliknutí kdekoľvek do poľa
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.active = True
            self.blink_timer = time.time()

            # Scrollbar drag
            lines = self._get_lines()
            line_h = self.font.get_height()
            total_h = len(lines) * line_h + 10
            inner = self.rect.inflate(-20, -20); inner.y += 10; inner.height -= 20
            if total_h > inner.height:
                max_s = total_h - inner.height
                bar_h = max(20, int(inner.height * inner.height / total_h))
                bar_y = inner.y + (self.scroll_y / max_s * (inner.height - bar_h)) if max_s > 0 else inner.y
                bar_rect = pygame.Rect(inner.right + 5, bar_y, 8, bar_h)
                if bar_rect.collidepoint(event.pos):
                    self.dragging_scroll = True
                    return

            # Kliknutie do textu
            mx, my = event.pos
            rel_y = my - inner.y + self.scroll_y
            row = int(rel_y // line_h)
            row = max(0, min(row, len(lines)-1))
            line = lines[row]
            x = 0
            col = 0
            for i, ch in enumerate(line + ' '):
                if inner.x + self.font.size(line[:i])[0] >= mx:
                    col = i
                    break
            idx = self._pos_to_char(row, col)

            if pygame.key.get_mods() & pygame.KMOD_SHIFT and self.select_start is not None:
                self.caret = idx
            else:
                self.select_start = idx
                self.caret = idx

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging_scroll = False

        elif event.type == pygame.MOUSEMOTION and self.dragging_scroll:
            lines = self._get_lines()
            line_h = self.font.get_height()
            total_h = len(lines) * line_h + 10
            inner = self.rect.inflate(-20, -20); inner.y += 10; inner.height -= 20
            if total_h > inner.height:
                max_s = total_h - inner.height
                bar_h = max(20, int(inner.height * inner.height / total_h))
                rel_y = event.pos[1] - inner.y
                ratio = rel_y / (inner.height - bar_h)
                self.scroll_y = max(0, min(int(ratio * max_s), max_s))

        # Koliesko myši (funguje aj keď nie je aktívny focus)
        elif event.type == pygame.MOUSEWHEEL and self.rect.collidepoint(pygame.mouse.get_pos()):
            self.scroll_y -= event.y * 30
            self.scroll_y = max(0, self.scroll_y)

        if not self.active:
            return

        if event.type == pygame.KEYDOWN:
            self.blink_timer = time.time()

            # Ctrl+A, Ctrl+C, Ctrl+V
            if event.key == pygame.K_a and event.mod & pygame.KMOD_CTRL:
                self.select_start = 0
                self.caret = len(self.text)
            elif event.key == pygame.K_c and event.mod & pygame.KMOD_CTRL:
                if self.select_start is not None:
                    s, e = sorted([self.select_start, self.caret])
                    pyperclip.copy(self.text[s:e])
            elif event.key == pygame.K_v and event.mod & pygame.KMOD_CTRL:
                paste = pyperclip.paste()
                s = e = self.caret
                if self.select_start is not None:
                    s, e = sorted([self.select_start, self.caret])
                self.text = self.text[:s] + paste + self.text[e:]
                self.caret = s + len(paste)
                self.select_start = None

            # Backspace / Delete
            elif event.key == pygame.K_BACKSPACE:
                if self.select_start is not None:
                    s, e = sorted([self.select_start, self.caret])
                    self.text = self.text[:s] + self.text[e:]
                    self.caret = s
                    self.select_start = None
                elif self.caret > 0:
                    self.text = self.text[:self.caret-1] + self.text[self.caret:]
                    self.caret -= 1

            elif event.key == pygame.K_DELETE:
                if self.select_start is not None:
                    s, e = sorted([self.select_start, self.caret])
                    self.text = self.text[:s] + self.text[e:]
                    self.caret = s
                    self.select_start = None
                elif self.caret < len(self.text):
                    self.text = self.text[:self.caret] + self.text[self.caret+1:]

            # Šípky – fungujú perfektne
            elif event.key == pygame.K_LEFT:
                if self.caret > 0: self.caret -= 1
                if not (event.mod & pygame.KMOD_SHIFT): self.select_start = None
            elif event.key == pygame.K_RIGHT:
                if self.caret < len(self.text): self.caret += 1
                if not (event.mod & pygame.KMOD_SHIFT): self.select_start = None
            elif event.key == pygame.K_UP:
                r, c = self._char_to_pos(self.caret)
                if r > 0: self.caret = self._pos_to_char(r-1, c)
                if not (event.mod & pygame.KMOD_SHIFT): self.select_start = None
            elif event.key == pygame.K_DOWN:
                r, c = self._char_to_pos(self.caret)
                lines = self._get_lines()
                if r < len(lines)-1: self.caret = self._pos_to_char(r+1, c)
                if not (event.mod & pygame.KMOD_SHIFT): self.select_start = None

            # Normálne písanie
            elif event.unicode:
                s = e = self.caret
                if self.select_start is not None:
                    s, e = sorted([self.select_start, self.caret])
                self.text = self.text[:s] + event.unicode + self.text[e:]
                self.caret = s + 1
                self.select_start = None

            # AUTOMATICKÉ SCROLLOVANIE NA KURZOR (najdôležitejšie!)
            row, _ = self._char_to_pos(self.caret)
            line_h = self.font.get_height()
            cursor_y = row * line_h
            inner = self.rect.inflate(-20, -20)
            inner.y += 10; inner.height -= 20

            if cursor_y - self.scroll_y > inner.height - line_h:
                self.scroll_y = cursor_y - inner.height + line_h + 20
            elif cursor_y - self.scroll_y < 0:
                self.scroll_y = cursor_y

            total_h = len(self._get_lines()) * line_h + 10
            if total_h > inner.height:
                max_s = total_h - inner.height
                self.scroll_y = max(0, min(self.scroll_y, max_s))
            else:
                self.scroll_y = 0

    def draw(self, screen):
        pygame.draw.rect(screen, CARD, self.rect, border_radius=12)
        pygame.draw.rect(screen, ACC if self.active else (70, 70, 80), self.rect, width=2, border_radius=12)

        inner = self.rect.inflate(-20, -20)
        inner.y += 10
        inner.height -= 20
        screen.set_clip(inner)

        # BG bude rovnaké ako CARD → žiadny rozdiel medzi vonkajšou a vnútornou farbou
        bg = CARD

        pygame.draw.rect(screen, bg, inner)

        lines = self._get_lines()
        line_h = self.font.get_height()
        total_h = len(lines) * line_h + 10
        if total_h > inner.height:
            max_s = total_h - inner.height
            self.scroll_y = max(0, min(self.scroll_y, max_s))
        else:
            self.scroll_y = 0

        # Modrý výber (viditeľný text)
        if self.select_start is not None:
            s, e = sorted([self.select_start, self.caret])
            pos = 0
            for row, line in enumerate(lines):
                start_char = max(pos, s) - pos
                end_char = min(pos + len(line), e) - pos
                if start_char < end_char:
                    x = inner.x + self.font.size(line[:start_char])[0]
                    w = self.font.size(line[start_char:end_char])[0]
                    y = inner.y + row * line_h - self.scroll_y
                    pygame.draw.rect(screen, (100, 160, 255, 180), (x, y, w, line_h))
                pos += len(line) + 1

        # Text + placeholder
        for i, line in enumerate(lines):
            y = inner.y + i * line_h - self.scroll_y
            color = TXT if line else MUTED
            text_to_draw = line if line else self.placeholder
            screen.blit(self.font.render(text_to_draw, True, color), (inner.x, y))

        # Blikajúci kurzor
        if self.active and (time.time() - self.blink_timer < 10) and (time.time() % 1.0 < 0.5):
            row, col = self._char_to_pos(self.caret)
            if row < len(lines):
                x = inner.x + self.font.size(lines[row][:col])[0]
                y = inner.y + row * line_h - self.scroll_y
                pygame.draw.line(screen, ACC, (x, y), (x, y + line_h - 4), 2)

        screen.set_clip(None)

        # Scrollbar
        if total_h > inner.height:
            bar_h = max(20, int(inner.height * inner.height / total_h))
            bar_y = inner.y + (self.scroll_y / max_s * (inner.height - bar_h)) if max_s > 0 else inner.y
            bar = pygame.Rect(inner.right + 5, bar_y, 8, bar_h)
            pygame.draw.rect(screen, (100,100,120), bar, border_radius=4)
            pygame.draw.rect(screen, (150,150,170), bar, width=1, border_radius=4)

# ----------------------------- Avatar -----------------------------
class Avatar:
    def __init__(self):
        self.base = load_img('base_face.png')
        self.eyes = {
            'Neutral': load_img('natural_eyes.png'),
            'Happy': load_img('happy_eyes.png'),
            'Surprised': load_img('surprised_eyes.png')
        }
        self.mouths = {
            'closed': load_img('closed_mouth.png'),
            'smile': load_img('smile_mouth.png'),
            'teeth': load_img('teeth_mouth.png'),
            'open': load_img('open_mouth.png'),
            'o': load_img('o_mouth.png'),
            'happy': load_img('happy_mouth.png'),
            'surprised': load_img('surprised_mouth.png')
        }
        self.emotion = 'Neutral';
        self.mouth = 'closed';
        self.native_size = self.base.get_size()

        # --- Idle animácia (blikanie) ---
        self.blink = False
        self.blink_timer = 0.0
        self.idle_timer = 0.0
        self.blink_duration = 0.1  # 100 ms bliknutia

    def set_emotion(self, e):
        self.emotion = e

    def set_mouth(self, m):
        self.mouth = m

    def update_idle(self, dt):
        if self.blink:
            self.blink_timer += dt
            if self.blink_timer >= self.blink_duration:
                self.blink = False
                self.blink_timer = 0.0
        else:
            self.idle_timer += dt
            if self.idle_timer > random.uniform(3.0, 6.0):  # blikne každých 3-6 sekúnd
                self.blink = True
                self.idle_timer = 0.0

    def draw_scaled(self, surf, rect: pygame.Rect):
        pad = 12;
        aw, ah = rect.width - 2 * pad, rect.height - 2 * pad;
        bw, bh = self.native_size
        s = min(aw / bw, ah / bh);
        nw, nh = int(bw * s), int(bh * s)
        x = rect.x + (rect.width - nw) // 2;
        y = rect.y + (rect.height - nh) // 2
        base = pygame.transform.smoothscale(self.base, (nw, nh))

        # --- Blikanie: použi happy oči ako zatvorené ---
        eye_key = 'Happy' if self.blink else self.emotion
        eyes = pygame.transform.smoothscale(self.eyes[eye_key], (nw, nh))

        mouth_key = self.mouth
        if self.emotion == 'Happy' and self.mouth in ('closed', 'smile'): mouth_key = 'smile'
        if self.emotion == 'Surprised' and self.mouth in ('open', 'o', 'teeth'): mouth_key = 'surprised'
        mouth = pygame.transform.smoothscale(self.mouths[mouth_key], (nw, nh))

        surf.blit(base, (x, y));
        surf.blit(eyes, (x, y));
        surf.blit(mouth, (x, y))


# ----------------------------- Prehrávač s lip-sync -----------------------------
class SpeechPlayer:
    def __init__(self):
        self.playing = False;
        self.start_t = 0.0;
        self.rms = [];
        self.rms_s = 0.0;
        self.wav = None;
        self.duration = 0.0
        pygame.mixer.init(frequency=22050, channels=2)

    def load_and_play(self, wav_path: str, volume: float = 0.9):
        self.stop();
        self.wav = wav_path
        self.rms, _, self.duration = wav_rms_timeline(wav_path)
        self.rms_s = 0.0;
        pygame.mixer.music.load(wav_path);
        pygame.mixer.music.set_volume(volume);
        pygame.mixer.music.play()
        self.start_t = time.time();
        self.playing = True

    def stop(self):
        if self.playing: pygame.mixer.music.stop()
        self.playing = False;
        self.rms = []
        if self.wav and os.path.exists(self.wav):
            try:
                os.remove(self.wav)
            except Exception:
                pass
        self.wav = None

    def progress(self) -> float:
        if not self.playing or self.duration <= 0: return 0.0
        p = (time.time() - self.start_t) / self.duration
        return max(0.0, min(1.0, p))

    def current_rms(self) -> float:
        if not self.playing or not self.rms: return 0.0
        idx = int(((time.time() - self.start_t) * 1000.0) // FRAME_MS)
        if idx >= len(self.rms): self.stop(); return 0.0
        inst = self.rms[idx];
        self.rms_s = RMS_SMOOTH * self.rms_s + (1.0 - RMS_SMOOTH) * inst
        return self.rms_s


# ----------------------------- App -----------------------------
class App:
    def __init__(self):
        pygame.init()
        pygame.key.set_repeat(400, 30)
        pygame.display.set_caption('Hovoriaci avatar – offline')
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.font = pygame.font.Font(FONT_NAME, 20)
        self.font_s = pygame.font.Font(FONT_NAME, 14)
        self.font_t = pygame.font.Font(FONT_NAME, 24)
        self.avatar = Avatar()
        self.player = SpeechPlayer()

        # === Rozloženie panelov ===
        # Ľavý panel – avatar
        self.left_rect = pygame.Rect(0, 0, LEFT_W, WINDOW_H)

        # Pravý panel – ovládanie
        self.right_rect = pygame.Rect(LEFT_W, 0, RIGHT_W, WINDOW_H)

        # === Prvky na pravej strane ===
        pad = PAD
        y = pad + 10

        # Prepínač témy (vpravo hore)
        self.btn_theme = Button(
            pygame.Rect(LEFT_W + RIGHT_W - 100 - pad, y, 100, 30),
            'Svetlý' if CONFIG['theme'] == 'dark' else 'Tmavý',
            self.toggle_theme
        )
        y += 50

        # Namiesto starého TextBox
        self.box = MultiLineTextBox(
            rect=(0, 0, 100, 100),  # veľkosť sa prepíše v run()
            font=self.font,
            placeholder='Napíš text…'
        )

        # Tlačidlá Speak / Stop
        half = (RIGHT_W - 3 * pad) // 2
        self.btn_speak = Button((LEFT_W + pad, y, half, CTRL_H), 'Speak', self.on_speak)
        self.btn_stop = Button((LEFT_W + pad + half + pad, y, half, CTRL_H), 'Stop', self.on_stop)
        y += CTRL_H + 20

        # Dropdown – jazyk
        self.tgl_lang = Dropdown(
            pygame.Rect(LEFT_W + pad, y, RIGHT_W - 2 * pad, CTRL_H + SMALL_H),
            ['SK', 'EN'], 0, 'Jazyk'
        )

        # Ostatné
        self.emotion_windows = []
        self.base_emo = 'Neutral'
        self.words = []

    def toggle_theme(self):
        # Prepni tému a ulož
        CONFIG['theme'] = 'light' if CONFIG['theme'] == 'dark' else 'dark'
        apply_theme(CONFIG['theme'])
        save_config(CONFIG)
        # Aktualizuj label tlačidla
        self.btn_theme.label = 'Svetlý' if CONFIG['theme'] == 'dark' else 'Tmavý'

    def on_speak(self):
        text = self.box.text.strip()
        if not text: return
       # print(f"[DEBUG] Text: {text}")  # <--- PRIDAJ
        wav = espeak_wav(text, self.tgl_lang.value, ESPEAK_WPM)
        self.emotion_windows = find_emotion_windows(text)
        #print(f"[DEBUG] Emócie okná: {self.emotion_windows}")  # <--- PRIDAJ
        self.base_emo = base_emotion(text)
       # print(f"[DEBUG] Base emócia: {self.base_emo}")  # <--- PRIDAJ
        self.last_emotion = None
        self.player.load_and_play(wav, volume=0.9)

    def on_stop(self):
        self.player.stop()
        self.speech_end_time = None  # ← toto pridaj
        self.last_active_emo = "Neutral"

    def draw_meter(self, rms: float):
        v = min(rms / RMS_METER_CLAMP, 1.0)
        h = 80
        w = 10
        x = LEFT_W + RIGHT_W - PAD - w - 6
        y = PAD + 48
        pygame.draw.rect(self.screen, (55, 58, 66), (x, y, w, h), border_radius=4)
        pygame.draw.rect(self.screen, ACC, (x, y + h - int(h * v), w, int(h * v)), border_radius=4)

    # 2. Ktorá emócia je práve aktívna – presne podľa času
    def current_emotion(self) -> str:
        # Ak práve hrá reč
        if self.player.playing:
            progress = self.player.progress()
            active_emo = self.base_emo

            # Nájdi aktuálne emočné okno
            for start, end, emo in self.emotion_windows:
                if start <= progress <= end:
                    active_emo = emo
                    break

            # Zapamätaj si poslednú emóciu a čas, kedy skončila reč
            self.last_active_emo = active_emo
            self.speech_end_time = None  # reč stále beží
            return active_emo

        # Ak reč skončila
        else:
            # Ak sme prvýkrát skončili → zapamätaj čas
            if not hasattr(self, 'speech_end_time') or self.speech_end_time is None:
                self.speech_end_time = time.time()

            # Po 2.5 sekundách od skončenia reči → vráť sa na Neutral
            if time.time() - self.speech_end_time > 2.5:
                return "Neutral"
            else:
                # Inak drž poslednú emóciu ešte chvíľu
                return getattr(self, "last_active_emo", "Neutral")

    def run(self):
        clock = pygame.time.Clock()
        running = True
        dt = 0.0

        # Trvalé rozloženie
        PAD = 24
        RIGHT_W = 440
        RIGHT_X = WINDOW_W - RIGHT_W - PAD
        LEFT_W = RIGHT_X - PAD * 2
        LEFT_X = PAD

        while running:
            dt = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN and self.box.active:
                    self.on_speak()

                self.box.handle(event)
                self.btn_speak.handle(event)
                self.btn_stop.handle(event)
                self.tgl_lang.handle(event)
                self.btn_theme.handle(event)

            self.screen.fill(BG)

            # Ľavý panel – Naruto
            avatar_rect = pygame.Rect(LEFT_X, 40, LEFT_W, WINDOW_H - 80)
            pygame.draw.rect(self.screen, CARD, avatar_rect, border_radius=20)
            pygame.draw.rect(self.screen, (60, 60, 80), avatar_rect, width=3, border_radius=20)

            # Lip-sync a emócie
            rms = self.player.current_rms()
            state = "closed"
            if rms > RMS_THR4:
                state = "o"
            elif rms > RMS_THR3:
                state = "open"
            elif rms > RMS_THR2:
                state = "teeth"
            emo = self.current_emotion()
            if emo == "Happy" and rms < RMS_THR2: state = "smile"
            self.avatar.set_emotion(emo)
            self.avatar.set_mouth(state)
            self.avatar.update_idle(dt)
            self.avatar.draw_scaled(self.screen, avatar_rect.inflate(-40, -40))

            # --- Postupný návrat na Neutral po skončení reči ---
            if not self.player.playing and hasattr(self, 'last_emotion') and self.last_emotion != 'Neutral':
                if not hasattr(self, 'neutral_timer'):
                    self.neutral_timer = time.time()
                elif time.time() - self.neutral_timer > 2.0:
                    self.last_emotion = 'Neutral'
                    del self.neutral_timer
            else:
                if hasattr(self, 'neutral_timer'):
                    del self.neutral_timer

            y = 36

            # Téma – úplne zarovno s textovým poľom
            box_x = RIGHT_X + 20
            box_w = RIGHT_W - 40

            self.btn_theme.rect = pygame.Rect(box_x, y, box_w, 40)
            self.btn_theme.draw(self.screen, self.font_s)
            y += 76

            # Textové pole
            box_h = 256
            self.box.rect = pygame.Rect(RIGHT_X + 20, y, RIGHT_W - 40, box_h)
            self.box.draw(self.screen)
            y += box_h + 28

            # Speak + Stop – ploché, bez pozadia
            btn_w = (RIGHT_W - 60) // 2
            self.btn_speak.rect = pygame.Rect(RIGHT_X + 20, y, btn_w, 52)
            self.btn_stop.rect = pygame.Rect(RIGHT_X + 20 + btn_w + 20, y, btn_w, 52)

            # Oprav tlačidlá – nech nemajú svetlé pozadie!
            def draw_clean_button(btn):
                color = ACC if btn.rect.collidepoint(pygame.mouse.get_pos()) else (80, 80, 100)
                pygame.draw.rect(self.screen, color, btn.rect, border_radius=12)
                pygame.draw.rect(self.screen, ACC, btn.rect, width=2, border_radius=12)
                txt = self.font.render(btn.label, True, TXT)
                self.screen.blit(txt, txt.get_rect(center=btn.rect.center))

            draw_clean_button(self.btn_speak)
            draw_clean_button(self.btn_stop)
            y += 70

            # Jazyk
            self.tgl_lang.rect = pygame.Rect(RIGHT_X + 20, y, RIGHT_W - 40, 66)
            self.tgl_lang.draw(self.screen, self.font, self.font_s)

            # Meter hlasitosti – vždy viditeľný
            mx = RIGHT_X + RIGHT_W - 508
            my = WINDOW_H - 560
            fill_h = int(110 * min(rms / RMS_METER_CLAMP, 1.0)) if self.player.playing else 0
            # Pozadie (vždy vidno)
            pygame.draw.rect(self.screen, (50, 50, 70), (mx, my, 16, 110), border_radius=8)
            # Výplň (iba keď hrá)
            if fill_h > 0:
                pygame.draw.rect(self.screen, ACC, (mx, my + 110 - fill_h, 16, fill_h), border_radius=8)
                # Rámik
            pygame.draw.rect(self.screen, (100, 100, 140), (mx, my, 16, 110), width=2, border_radius=8)
            pygame.display.flip()

        self.player.stop()
        pygame.quit()


if __name__ == '__main__':
    try:
        App().run()
    except FileNotFoundError as e:
        print('\n[CHÝBA ASSET]', e)
        print('Skontroluj ./assets názvy súborov.')
    except Exception as ex:
        print('Chyba:', ex)
