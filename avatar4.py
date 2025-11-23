import os, tempfile, wave, time, re, json, random
from typing import List, Tuple

import numpy as np
import pygame
import pyperclip
from piper import PiperVoice

# ----------------------------- Konštanty -----------------------------
BASE_DIR   = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
VOICES_DIR = os.path.join(BASE_DIR, "voices")
CONFIG_PATH = 'config.json'  # pre ukladanie témy

# Okno
WINDOW_W = 960          # 2× širšie okno
WINDOW_H = 620          # nižšie, UI vpravo

# Predvolené farby (tmavý režim)
BG = (16, 18, 22)
PANEL = (26, 28, 34)
CARD = (32, 35, 42)
TXT = (230, 230, 235)
MUTED = (165, 170, 180)
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

# Layout
LEFT_W = 480            # ľavý panel = avatar
RIGHT_W = 480           # pravý panel = UI
PAD = 16
GAP = 10
CTRL_H = 44
SMALL_H = 18

LEFT_X  = PAD
RIGHT_X = LEFT_W + PAD

# ----------------------------- Piper TTS – inicializácia -----------------------------
# SK Lili
PIPER_SK_MODEL_ONNX = os.path.join(VOICES_DIR, "sk_SK-lili-medium.onnx")
PIPER_SK_MODEL_JSON = os.path.join(VOICES_DIR, "sk_SK-lili-medium.onnx.json")

if not (os.path.exists(PIPER_SK_MODEL_ONNX) and os.path.exists(PIPER_SK_MODEL_JSON)):
    raise FileNotFoundError(
        f"Chýba SK Piper model v priečinku ./voices. "
        f"Očakávam: {PIPER_SK_MODEL_ONNX} a {PIPER_SK_MODEL_JSON}"
    )

print("[piper] [info] Loading SK voice (Lili)…")
PIPER_SK_VOICE = PiperVoice.load(PIPER_SK_MODEL_ONNX, PIPER_SK_MODEL_JSON)

# EN Amy – ak chýba, budeme fallbackovať na SK
PIPER_EN_MODEL_ONNX = os.path.join(VOICES_DIR, "en_US-amy-medium.onnx")
PIPER_EN_MODEL_JSON = os.path.join(VOICES_DIR, "en_US-amy-medium.onnx.json")
if os.path.exists(PIPER_EN_MODEL_ONNX) and os.path.exists(PIPER_EN_MODEL_JSON):
    print("[piper] [info] Loading EN voice (Amy)…")
    PIPER_EN_VOICE = PiperVoice.load(PIPER_EN_MODEL_ONNX, PIPER_EN_MODEL_JSON)
    HAS_EN_VOICE = True
else:
    print("[piper] [warn] EN voice not found, using SK Lili also for EN.")
    PIPER_EN_VOICE = None
    HAS_EN_VOICE = False


def piper_tts_raw(text: str, lang: str) -> str:
    """
    Vygeneruje základný WAV z Piper hlasu (bez úprav).
    Používa synthesize_wav → wave.open s nastavenými parametrami.
    Podľa lang ('SK'/'EN') vyberie hlas.
    """
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    with wave.open(wav_path, "wb") as wav_file:
        wav_file.setnchannels(1)      # mono
        wav_file.setsampwidth(2)      # 16-bit
        wav_file.setframerate(22050)  # 22.05 kHz

        if lang == "EN" and HAS_EN_VOICE:
            voice = PIPER_EN_VOICE
        else:
            voice = PIPER_SK_VOICE

        voice.synthesize_wav(text, wav_file)

    return wav_path


def _pitch_shift(data: np.ndarray, semitones: float) -> np.ndarray:
    """Jednoduchý pitch shift pomocou resamplingu (zmení aj tempo)."""
    factor = 2 ** (semitones / 12.0)
    idx = np.arange(0, len(data), 1.0 / factor)
    idx = idx[idx < len(data)]
    return np.interp(idx, np.arange(len(data)), data)


def _speed_change(data: np.ndarray, speed_factor: float) -> np.ndarray:
    """Zrýchlenie/spomalenie signálu výberom vzoriek."""
    idx = np.arange(0, len(data), speed_factor)
    idx = idx[idx < len(data)]
    return data[idx.astype(int)]


def piper_tts_anime(text: str, lang: str) -> str:
    """
    Text -> "anime" hlas (Naruto-ish).
    SK: Lili + vyšší pitch a rýchlosť
    EN: Amy + jemnejší pitch/speed
    """
    base_path = piper_tts_raw(text, lang)

    # načítaj WAV cez wave + numpy
    with wave.open(base_path, "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)

    if n == 0:
        return base_path

    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        return base_path

    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)

    if lang == "EN":
        semitones = 1.4   # jemne vyšší hlas
        speed = 1.06      # mierne rýchlejšie
    else:
        semitones = 2.8   # výraznejší anime efekt
        speed = 1.12

    data = _pitch_shift(data, semitones)
    data = _speed_change(data, speed)
    data = data / (np.max(np.abs(data)) + 1e-6)

    data_i16 = (data * 32767.0).astype(np.int16)

    out_fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data_i16.tobytes())

    try:
        os.remove(base_path)
    except OSError:
        pass

    return out_path

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
        BG = (16, 18, 22)
        PANEL = (26, 28, 34)
        CARD = (32, 35, 42)
        TXT = (230, 230, 235)
        MUTED = (165, 170, 180)
        ACC = (120, 180, 255)
    else:  # light
        BG = (240, 240, 245)
        PANEL = (220, 220, 230)
        CARD = (255, 255, 255)
        TXT = (30, 30, 35)
        MUTED = (100, 100, 110)
        ACC = (0, 120, 255)


apply_theme(CONFIG["theme"])

# ----------------------------- Utility -----------------------------
def load_img(name: str) -> pygame.Surface:
    p = os.path.join(ASSETS_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f'Chýba súbor: {p}')
    return pygame.image.load(p).convert_alpha()


def wav_rms_timeline(path: str, frame_ms: int = FRAME_MS) -> Tuple[List[float], int, float]:
    with wave.open(path, 'rb') as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
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


def find_emotion_windows(text: str) -> List[Tuple[float, float, str]]:
    t = text.lower()
    tokens = _word_re.findall(t)
    if not tokens:
        return []

    total_words = len(tokens)
    windows = []

    for i, word in enumerate(tokens):
        word_clean = word.replace(',', '').replace('.', '').replace('!', '').replace('?', '')
        position = i / total_words

        emo = None
        duration = 0.22

        if any(k in word_clean for k in [
            "wow", "čože", "coze", "fíha", "fiha", "vážne", "vazne", "neverím", "neverim", "prekvap", "!?"]):
            emo = "Surprised"
            duration = 0.35
        elif any(k in word_clean for k in [
            "teším sa", "tesim sa", "super", "skvelé", "paráda", "parada",
            "milujem", "rád", "rad", "jéé", "jupí", "hurá"]):
            emo = "Happy"

        if emo:
            start = position
            end = min(1.0, position + duration)
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
        self.rect = pygame.Rect(rect)
        self.label = label
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
        lab = font_small.render(self.label, True, MUTED)
        surf.blit(lab, (self.rect.x + 2, self.rect.y))

        main_box = pygame.Rect(self.rect.x, self.rect.y + SMALL_H, self.rect.width, CTRL_H)
        pygame.draw.rect(surf, CARD, main_box, border_radius=8)
        pygame.draw.rect(surf, ACC if self.open else (70, 70, 80), main_box, width=2, border_radius=8)
        surf.blit(font.render(self.value, True, TXT), (main_box.x + 12, main_box.y + 10))

        arrow_x = main_box.right - 30
        arrow_y = main_box.centery
        points = [(arrow_x, arrow_y - 5), (arrow_x + 10, arrow_y - 5), (arrow_x + 5, arrow_y + 3)]
        pygame.draw.polygon(surf, TXT if self.open else MUTED, points)

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

# ----------------------------- TextBox (multiline) -----------------------------
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
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.active = True
            self.blink_timer = time.time()

            lines = self._get_lines()
            line_h = self.font.get_height()
            total_h = len(lines) * line_h + 10
            inner = self.rect.inflate(-20, -20)
            inner.y += 10
            inner.height -= 20
            if total_h > inner.height:
                max_s = total_h - inner.height
                bar_h = max(20, int(inner.height * inner.height / total_h))
                bar_y = inner.y + (self.scroll_y / max_s * (inner.height - bar_h)) if max_s > 0 else inner.y
                bar_rect = pygame.Rect(inner.right + 5, bar_y, 8, bar_h)
                if bar_rect.collidepoint(event.pos):
                    self.dragging_scroll = True
                    return

            mx, my = event.pos
            rel_y = my - inner.y + self.scroll_y
            row = int(rel_y // line_h)
            row = max(0, min(row, len(lines)-1))
            line = lines[row]
            col = 0
            for i in range(len(line)+1):
                w = self.font.size(line[:i])[0]
                if inner.x + w >= mx:
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
            inner = self.rect.inflate(-20, -20)
            inner.y += 10
            inner.height -= 20
            if total_h > inner.height:
                max_s = total_h - inner.height
                bar_h = max(20, int(inner.height * inner.height / total_h))
                rel_y = event.pos[1] - inner.y
                ratio = rel_y / (inner.height - bar_h)
                self.scroll_y = max(0, min(int(ratio * max_s), max_s))

        elif event.type == pygame.MOUSEWHEEL and self.rect.collidepoint(pygame.mouse.get_pos()):
            self.scroll_y -= event.y * 30
            self.scroll_y = max(0, self.scroll_y)

        if not self.active:
            return

        if event.type == pygame.KEYDOWN:
            self.blink_timer = time.time()

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

            elif event.key == pygame.K_LEFT:
                if self.caret > 0:
                    self.caret -= 1
                if not (event.mod & pygame.KMOD_SHIFT):
                    self.select_start = None
            elif event.key == pygame.K_RIGHT:
                if self.caret < len(self.text):
                    self.caret += 1
                if not (event.mod & pygame.KMOD_SHIFT):
                    self.select_start = None
            elif event.key == pygame.K_UP:
                r, c = self._char_to_pos(self.caret)
                if r > 0:
                    self.caret = self._pos_to_char(r-1, c)
                if not (event.mod & pygame.KMOD_SHIFT):
                    self.select_start = None
            elif event.key == pygame.K_DOWN:
                r, c = self._char_to_pos(self.caret)
                lines = self._get_lines()
                if r < len(lines)-1:
                    self.caret = self._pos_to_char(r+1, c)
                if not (event.mod & pygame.KMOD_SHIFT):
                    self.select_start = None
            elif event.unicode:
                s = e = self.caret
                if self.select_start is not None:
                    s, e = sorted([self.select_start, self.caret])
                self.text = self.text[:s] + event.unicode + self.text[e:]
                self.caret = s + 1
                self.select_start = None

            row, _ = self._char_to_pos(self.caret)
            line_h = self.font.get_height()
            cursor_y = row * line_h
            inner = self.rect.inflate(-20, -20)
            inner.y += 10
            inner.height -= 20

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
        pygame.draw.rect(screen, ACC if self.active else (70, 70, 80),
                         self.rect, width=2, border_radius=12)

        inner = self.rect.inflate(-20, -20)
        inner.y += 10
        inner.height -= 20
        screen.set_clip(inner)

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
                    pygame.draw.rect(screen, (100, 160, 255, 180),
                                     (x, y, w, line_h))
                pos += len(line) + 1

        for i, line in enumerate(lines):
            y = inner.y + i * line_h - self.scroll_y
            color = TXT if line else MUTED
            text_to_draw = line if line else self.placeholder
            screen.blit(self.font.render(text_to_draw, True, color),
                        (inner.x, y))

        if self.active and (time.time() - self.blink_timer < 10) and (time.time() % 1.0 < 0.5):
            row, col = self._char_to_pos(self.caret)
            if row < len(lines):
                x = inner.x + self.font.size(lines[row][:col])[0]
                y = inner.y + row * line_h - self.scroll_y
                pygame.draw.line(screen, ACC, (x, y),
                                 (x, y + line_h - 4), 2)

        screen.set_clip(None)

        total_h = len(lines) * line_h + 10
        inner = self.rect.inflate(-20, -20)
        inner.y += 10
        inner.height -= 20
        if total_h > inner.height:
            max_s = total_h - inner.height
            bar_h = max(20, int(inner.height * inner.height / total_h))
            bar_y = inner.y + (self.scroll_y / max_s *
                               (inner.height - bar_h)) if max_s > 0 else inner.y
            bar = pygame.Rect(inner.right + 5, bar_y, 8, bar_h)
            pygame.draw.rect(screen, (100, 100, 120), bar, border_radius=4)
            pygame.draw.rect(screen, (150, 150, 170), bar,
                             width=1, border_radius=4)

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
        self.emotion = 'Neutral'
        self.mouth = 'closed'
        self.native_size = self.base.get_size()

        self.blink = False
        self.blink_timer = 0.0
        self.idle_timer = 0.0
        self.blink_duration = 0.1

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
            if self.idle_timer > random.uniform(3.0, 6.0):
                self.blink = True
                self.idle_timer = 0.0

    def draw_scaled(self, surf, rect: pygame.Rect):
        pad = 12
        aw, ah = rect.width - 2 * pad, rect.height - 2 * pad
        bw, bh = self.native_size
        s = min(aw / bw, ah / bh)
        nw, nh = int(bw * s), int(bh * s)
        x = rect.x + (rect.width - nw) // 2
        y = rect.y + (rect.height - nh) // 2
        base = pygame.transform.smoothscale(self.base, (nw, nh))

        eye_key = 'Happy' if self.blink else self.emotion
        eyes = pygame.transform.smoothscale(self.eyes[eye_key], (nw, nh))

        mouth_key = self.mouth
        if self.emotion == 'Happy' and self.mouth in ('closed', 'smile'):
            mouth_key = 'smile'
        if self.emotion == 'Surprised' and self.mouth in ('open', 'o', 'teeth'):
            mouth_key = 'surprised'
        mouth = pygame.transform.smoothscale(self.mouths[mouth_key], (nw, nh))

        surf.blit(base, (x, y))
        surf.blit(eyes, (x, y))
        surf.blit(mouth, (x, y))

# ----------------------------- Prehrávač s lip-sync -----------------------------
class SpeechPlayer:
    def __init__(self):
        self.playing = False
        self.start_t = 0.0
        self.rms = []
        self.rms_s = 0.0
        self.wav = None
        self.duration = 0.0
        pygame.mixer.init(frequency=22050, channels=2)

    def load_and_play(self, wav_path: str, volume: float = 0.9):
        self.stop()
        self.wav = wav_path
        self.rms, _, self.duration = wav_rms_timeline(wav_path)
        self.rms_s = 0.0
        pygame.mixer.music.load(wav_path)
        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.play()
        self.start_t = time.time()
        self.playing = True

    def stop(self):
        if self.playing:
            pygame.mixer.music.stop()
        self.playing = False
        self.rms = []
        if self.wav and os.path.exists(self.wav):
            try:
                os.remove(self.wav)
            except Exception:
                pass
        self.wav = None

    def progress(self) -> float:
        if not self.playing or self.duration <= 0:
            return 0.0
        p = (time.time() - self.start_t) / self.duration
        return max(0.0, min(1.0, p))

    def current_rms(self) -> float:
        if not self.playing or not self.rms:
            return 0.0
        idx = int(((time.time() - self.start_t) * 1000.0) // FRAME_MS)
        if idx >= len(self.rms):
            self.stop()
            return 0.0
        inst = self.rms[idx]
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

        self.left_rect = pygame.Rect(0, 0, LEFT_W, WINDOW_H)
        self.right_rect = pygame.Rect(LEFT_W, 0, RIGHT_W, WINDOW_H)

        pad = PAD
        y = pad + 10

        self.btn_theme = Button(
            pygame.Rect(LEFT_W + RIGHT_W - 100 - pad, y, 100, 30),
            'Svetlý' if CONFIG['theme'] == 'dark' else 'Tmavý',
            self.toggle_theme
        )
        y += 50

        self.box = MultiLineTextBox(
            rect=(0, 0, 100, 100),
            font=self.font,
            placeholder='Napíš text…'
        )

        half = (RIGHT_W - 3 * pad) // 2
        self.btn_speak = Button((LEFT_W + pad, y, half, CTRL_H), 'Speak', self.on_speak)
        self.btn_stop = Button((LEFT_W + pad + half + pad, y, half, CTRL_H), 'Stop', self.on_stop)
        y += CTRL_H + 20

        self.tgl_lang = Dropdown(
            pygame.Rect(LEFT_W + pad, y, RIGHT_W - 2 * pad, CTRL_H + SMALL_H),
            ['SK', 'EN'], 0, 'Jazyk'
        )

        self.emotion_windows = []
        self.base_emo = 'Neutral'
        self.words = []
        self.last_emotion = 'Neutral'
        self.speech_end_time = None
        self.last_active_emo = 'Neutral'

    def toggle_theme(self):
        CONFIG['theme'] = 'light' if CONFIG['theme'] == 'dark' else 'dark'
        apply_theme(CONFIG['theme'])
        save_config(CONFIG)
        self.btn_theme.label = 'Svetlý' if CONFIG['theme'] == 'dark' else 'Tmavý'

    def on_speak(self):
        text = self.box.text.strip()
        if not text:
            return

        wav = piper_tts_anime(text, self.tgl_lang.value)

        self.emotion_windows = find_emotion_windows(text)
        self.base_emo = base_emotion(text)
        self.last_emotion = None
        self.player.load_and_play(wav, volume=0.9)

    def on_stop(self):
        self.player.stop()
        self.speech_end_time = None
        self.last_active_emo = "Neutral"

    def current_emotion(self) -> str:
        if self.player.playing:
            progress = self.player.progress()
            active_emo = self.base_emo

            for start, end, emo in self.emotion_windows:
                if start <= progress <= end:
                    active_emo = emo
                    break

            self.last_active_emo = active_emo
            self.speech_end_time = None
            return active_emo
        else:
            if not hasattr(self, 'speech_end_time') or self.speech_end_time is None:
                self.speech_end_time = time.time()

            if time.time() - self.speech_end_time > 2.5:
                return "Neutral"
            else:
                return getattr(self, "last_active_emo", "Neutral")

    def run(self):
        clock = pygame.time.Clock()
        running = True

        PAD_L = 24
        RIGHT_W_LOC = 440
        RIGHT_X_LOC = WINDOW_W - RIGHT_W_LOC - PAD_L
        LEFT_W_LOC = RIGHT_X_LOC - PAD_L * 2
        LEFT_X_LOC = PAD_L

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

            avatar_rect = pygame.Rect(LEFT_X_LOC, 40, LEFT_W_LOC, WINDOW_H - 80)
            pygame.draw.rect(self.screen, CARD, avatar_rect, border_radius=20)
            pygame.draw.rect(self.screen, (60, 60, 80), avatar_rect, width=3, border_radius=20)

            rms = self.player.current_rms()
            state = "closed"
            if rms > RMS_THR4:
                state = "o"
            elif rms > RMS_THR3:
                state = "open"
            elif rms > RMS_THR2:
                state = "teeth"
            emo = self.current_emotion()
            if emo == "Happy" and rms < RMS_THR2:
                state = "smile"
            self.avatar.set_emotion(emo)
            self.avatar.set_mouth(state)
            self.avatar.update_idle(dt)
            self.avatar.draw_scaled(self.screen, avatar_rect.inflate(-40, -40))

            y = 36
            box_x = RIGHT_X_LOC + 20
            box_w = RIGHT_W_LOC - 40

            self.btn_theme.rect = pygame.Rect(box_x, y, box_w, 40)
            self.btn_theme.draw(self.screen, self.font_s)
            y += 76

            box_h = 256
            self.box.rect = pygame.Rect(RIGHT_X_LOC + 20, y, RIGHT_W_LOC - 40, box_h)
            self.box.draw(self.screen)
            y += box_h + 28

            btn_w = (RIGHT_W_LOC - 60) // 2
            self.btn_speak.rect = pygame.Rect(RIGHT_X_LOC + 20, y, btn_w, 52)
            self.btn_stop.rect = pygame.Rect(RIGHT_X_LOC + 20 + btn_w + 20, y, btn_w, 52)

            def draw_clean_button(btn):
                color = ACC if btn.rect.collidepoint(pygame.mouse.get_pos()) else (80, 80, 100)
                pygame.draw.rect(self.screen, color, btn.rect, border_radius=12)
                pygame.draw.rect(self.screen, ACC, btn.rect, width=2, border_radius=12)
                txt = self.font.render(btn.label, True, TXT)
                self.screen.blit(txt, txt.get_rect(center=btn.rect.center))

            draw_clean_button(self.btn_speak)
            draw_clean_button(self.btn_stop)
            y += 70

            self.tgl_lang.rect = pygame.Rect(RIGHT_X_LOC + 20, y, RIGHT_W_LOC - 40, 66)
            self.tgl_lang.draw(self.screen, self.font, self.font_s)

            mx = RIGHT_X_LOC + RIGHT_W_LOC - 508
            my = WINDOW_H - 560
            fill_h = int(110 * min(rms / RMS_METER_CLAMP, 1.0)) if self.player.playing else 0
            pygame.draw.rect(self.screen, (50, 50, 70), (mx, my, 16, 110), border_radius=8)
            if fill_h > 0:
                pygame.draw.rect(self.screen, ACC, (mx, my + 110 - fill_h, 16, fill_h), border_radius=8)
            pygame.draw.rect(self.screen, (100, 100, 140), (mx, my, 16, 110), width=2, border_radius=8)

            pygame.display.flip()

        self.player.stop()
        pygame.quit()


if __name__ == '__main__':
    try:
        App().run()
    except FileNotFoundError as e:
        print('\n[CHÝBA ASSET]', e)
        print('Skontroluj ./assets názvy súborov alebo ./voices model Piper.')
    except Exception as ex:
        print('Chyba:', ex)
