"""
Hovoriaci avatar – OFFLINE (pygame + eSpeak‑NG)
------------------------------------------------
Stabilná verzia s:
• dvojpanelovým UI (vľavo avatar, vpravo ovládanie),
• lip‑syncom cez RMS + vyhladzovanie a 5 tvarov úst,
• AUTO emóciami z textu (SK/EN),
• „emočné okná“,
• plne funkčným textovým poľom (Ctrl+A/C/V/X, výber myšou).

Požiadavky: pip install pygame numpy pyperclip + eSpeak-NG
"""

import os, subprocess, tempfile, wave, time, re
from typing import List, Tuple
import numpy as np
import pygame
import pyperclip  # <-- pre Ctrl+C/V/X

# ----------------------------- Konštanty -----------------------------
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')

WINDOW_W = 480
WINDOW_H = 740

BG = (16, 18, 22); PANEL = (26, 28, 34); CARD = (32, 35, 42); TXT = (230, 230, 235); MUTED = (165, 170, 180); ACC = (120, 180, 255)
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
ESPEAK_VOICE_EN = 'en+f3'
ESPEAK_WPM = 170

# Layout
RIGHT_W = 420; LEFT_W = WINDOW_W - RIGHT_W
PAD = 16; GAP = 10; CTRL_H = 44; SMALL_H = 18


# ----------------------------- Utility -----------------------------
def load_img(name: str) -> pygame.Surface:
    p = os.path.join(ASSETS_DIR, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f'Chýba súbor: {p}')
    return pygame.image.load(p).convert_alpha()


def espeak_wav(text: str, lang: str, wpm: int) -> str:
    voice = ESPEAK_VOICE_SK if lang == 'SK' else ESPEAK_VOICE_EN
    fd, wav_path = tempfile.mkstemp(suffix='.wav'); os.close(fd)
    cmd = ['espeak-ng', '-v', voice, '-s', str(wpm), '-w', wav_path, text]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError('espeak-ng nie je v PATH. Doinštaluj eSpeak NG.')
    return wav_path


def wav_rms_timeline(path: str, frame_ms: int = FRAME_MS) -> Tuple[List[float], int, float]:
    with wave.open(path, 'rb') as wf:
        ch = wf.getnchannels(); sw = wf.getsampwidth(); sr = wf.getframerate(); n = wf.getnframes()
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
    total = len(tokens)
    windows: List[Tuple[float, float, str]] = []
    for i, w in enumerate(tokens):
        pct = i / total
        if any(k.replace(' ', '') in w.replace(' ', '') for k in POS_WORDS):
            windows.append((max(0.0, pct - 0.05), min(1.0, pct + 0.05), 'Happy'))
        if any(k.replace(' ', '') in w.replace(' ', '') for k in SURP_WORDS):
            windows.append((max(0.0, pct - 0.05), min(1.0, pct + 0.05), 'Surprised'))
    return windows


def base_emotion(text: str) -> str:
    t = text.lower()
    if any(k.replace(' ', '') in t.replace(' ', '') for k in SURP_WORDS):
        return 'Surprised'
    if any(k.replace(' ', '') in t.replace(' ', '') for k in POS_WORDS):
        return 'Happy'
    return 'Neutral'


# ----------------------------- UI prvky -----------------------------
class Button:
    def __init__(self, rect, label, on_click):
        self.rect = pygame.Rect(rect); self.label = label; self.on_click = on_click

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

        # Šípka – otočí sa podľa smeru otvorenia
        arrow_x = main_box.right - 30
        arrow_y = main_box.centery
        if self.open:
            # Šípka hore (ak sa otvára hore)
            if hasattr(self, '_drop_up') and self._drop_up:
                points = [(arrow_x, arrow_y + 5), (arrow_x + 10, arrow_y + 5), (arrow_x + 5, arrow_y - 3)]
            else:
                points = [(arrow_x, arrow_y - 5), (arrow_x + 10, arrow_y - 5), (arrow_x + 5, arrow_y + 3)]
        else:
            points = [(arrow_x, arrow_y - 5), (arrow_x + 10, arrow_y - 5), (arrow_x + 5, arrow_y + 3)]
        pygame.draw.polygon(surf, TXT if self.open else MUTED, points)

        # Otvorená ponuka
        if self.open:
            drop_h = len(self.options) * CTRL_H
            screen_h = surf.get_height()

            # Skús dole
            drop_y = main_box.bottom + 2
            drop_up = False

            # Ak sa nezmestí dole → otvori hore
            if drop_y + drop_h > screen_h:
                drop_y = main_box.y - drop_h - 2
                drop_up = True

            self._drop_up = drop_up  # zapamätaj si smer
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
                # Výpočet indexu podľa y
                rel_y = event.pos[1] - self.dropdown_rect.y
                idx = rel_y // CTRL_H
                if 0 <= idx < len(self.options):
                    self.select(idx)
                return
            # Klik mimo → zatvor
            if self.open:
                self.open = False


# ----------------------------- TextBox s výberom -----------------------------
class TextBox:
    def __init__(self, rect, text=''):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.active = False
        self.caret = len(text)
        self.select_start = self.caret  # Začiatok výberu
        self.scroll_x = 0
        self._blink_t = time.time()
        self._blink_on = True

    def _get_selection(self):
        if self.select_start == self.caret:
            return None, None
        start = min(self.caret, self.select_start)
        end = max(self.caret, self.select_start)
        return start, end

    def _ensure_caret_visible(self, font):
        if not self.text:
            self.scroll_x = 0
            return
        pad = 10
        box_w = self.rect.width - 2 * pad
        caret_px = font.size(self.text[:self.caret])[0]
        if caret_px - self.scroll_x > box_w:
            self.scroll_x = caret_px - box_w
        elif caret_px - self.scroll_x < 0:
            self.scroll_x = caret_px
        if self.scroll_x < 0:
            self.scroll_x = 0

    def _draw_selection(self, surf, font, box):
        start, end = self._get_selection()
        if start is None or start == end:
            return
        x1 = font.size(self.text[:start])[0]
        x2 = font.size(self.text[:end])[0]
        sel_rect = pygame.Rect(
            box.x - self.scroll_x + x1,
            box.y,
            x2 - x1,
            font.get_height()
        )
        pygame.draw.rect(surf, (100, 150, 255, 100), sel_rect, border_radius=2)

    def draw(self, surf, font, placeholder='Napíš text…'):
        pygame.draw.rect(surf, CARD, self.rect, border_radius=8)
        pygame.draw.rect(surf, ACC if self.active else (70, 70, 80),
                         self.rect, width=2, border_radius=8)
        pad = 10
        box = pygame.Rect(self.rect.x + pad, self.rect.y + pad,
                          self.rect.width - 2 * pad, self.rect.height - 2 * pad)

        self._draw_selection(surf, font, box)

        txt = self.text if self.text else placeholder
        color = TXT if self.text else MUTED
        text_surf = font.render(txt, True, color)
        surf.set_clip(box)
        surf.blit(text_surf, (box.x - self.scroll_x, box.y))
        surf.set_clip(None)

        if self.active and self._blink_on and self.select_start == self.caret:
            now = time.time()
            if now - self._blink_t > 0.5:
                self._blink_on = not self._blink_on
                self._blink_t = now
            if self._blink_on:
                caret_px = font.size(self.text[:self.caret])[0]
                cx = box.x - self.scroll_x + caret_px
                cy1 = box.y
                cy2 = box.y + font.get_height()
                pygame.draw.line(surf, ACC, (cx, cy1), (cx, cy2), 2)

    def handle(self, event, font=None):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
                pad = 10
                rel_x = event.pos[0] - (self.rect.x + pad) + self.scroll_x
                idx = 0
                for i in range(len(self.text) + 1):
                    if font.size(self.text[:i])[0] >= rel_x:
                        idx = i
                        break
                else:
                    idx = len(self.text)

                mods = pygame.key.get_mods()
                if mods & pygame.KMOD_SHIFT and self.select_start != self.caret:
                    self.caret = idx
                else:
                    self.select_start = idx
                    self.caret = idx
                self._ensure_caret_visible(font)
                self._blink_t = time.time()
                self._blink_on = True
            else:
                self.active = False

        elif event.type == pygame.MOUSEMOTION and event.buttons[0] and self.active:
            pad = 10
            rel_x = event.pos[0] - (self.rect.x + pad) + self.scroll_x
            idx = 0
            for i in range(len(self.text) + 1):
                if font.size(self.text[:i])[0] >= rel_x:
                    idx = i
                    break
            else:
                idx = len(self.text)
            self.caret = idx
            self._ensure_caret_visible(font)

        if not self.active or not font:
            return

        if event.type == pygame.KEYDOWN:
            mods = pygame.key.get_mods()

            # Ctrl+A
            if event.key == pygame.K_a and (mods & pygame.KMOD_CTRL):
                self.select_start = 0
                self.caret = len(self.text)
                self._ensure_caret_visible(font)

            # Ctrl+C
            elif event.key == pygame.K_c and (mods & pygame.KMOD_CTRL):
                start, end = self._get_selection()
                if start is not None and start != end:
                    pyperclip.copy(self.text[start:end])

            # Ctrl+V
            elif event.key == pygame.K_v and (mods & pygame.KMOD_CTRL):
                clip_text = pyperclip.paste()
                if clip_text:
                    start, end = self._get_selection()
                    if start is not None and start != end:
                        self.text = self.text[:start] + clip_text + self.text[end:]
                        self.caret = start + len(clip_text)
                    else:
                        self.text = self.text[:self.caret] + clip_text + self.text[self.caret:]
                        self.caret += len(clip_text)
                    self.select_start = self.caret
                    self._ensure_caret_visible(font)

            # Ctrl+X
            elif event.key == pygame.K_x and (mods & pygame.KMOD_CTRL):
                start, end = self._get_selection()
                if start is not None and start != end:
                    pyperclip.copy(self.text[start:end])
                    self.text = self.text[:start] + self.text[end:]
                    self.caret = start
                    self.select_start = self.caret
                    self._ensure_caret_visible(font)

            # Backspace / Delete
            elif event.key == pygame.K_BACKSPACE:
                start, end = self._get_selection()
                if start is not None and start != end:
                    self.text = self.text[:start] + self.text[end:]
                    self.caret = start
                    self.select_start = self.caret
                elif self.caret > 0:
                    self.text = self.text[:self.caret - 1] + self.text[self.caret:]
                    self.caret -= 1
                self._ensure_caret_visible(font)

            elif event.key == pygame.K_DELETE:
                start, end = self._get_selection()
                if start is not None and start != end:
                    self.text = self.text[:start] + self.text[end:]
                    self.caret = start
                    self.select_start = self.caret
                elif self.caret < len(self.text):
                    self.text = self.text[:self.caret] + self.text[self.caret + 1:]
                self._ensure_caret_visible(font)

            # Šípky
            elif event.key == pygame.K_LEFT:
                if mods & pygame.KMOD_SHIFT:
                    self.caret = max(0, self.caret - 1)
                else:
                    self.caret = max(0, self.caret - 1)
                    self.select_start = self.caret  # <--- ZRUŠ VÝBER
                self._ensure_caret_visible(font)


            elif event.key == pygame.K_RIGHT:

                if mods & pygame.KMOD_SHIFT:

                    self.caret = min(len(self.text), self.caret + 1)

                else:

                    self.caret = min(len(self.text), self.caret + 1)

                    self.select_start = self.caret  # <--- ZRUŠ VÝBER

                self._ensure_caret_visible(font)

            elif event.key == pygame.K_HOME:
                self.caret = 0
                if not (mods & pygame.KMOD_SHIFT):
                    self.select_start = self.caret
                self._ensure_caret_visible(font)

            elif event.key == pygame.K_END:
                self.caret = len(self.text)
                if not (mods & pygame.KMOD_SHIFT):
                    self.select_start = self.caret
                self._ensure_caret_visible(font)

            # Normálne písanie
            else:
                if event.unicode:
                    start, end = self._get_selection()
                    if start is not None and start != end:
                        self.text = self.text[:start] + event.unicode + self.text[end:]
                        self.caret = start + 1
                    else:
                        self.text = self.text[:self.caret] + event.unicode + self.text[self.caret:]
                        self.caret += 1
                    self.select_start = self.caret
                    self._ensure_caret_visible(font)

            self._blink_t = time.time()
            self._blink_on = True


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
        self.emotion = 'Neutral'; self.mouth = 'closed'; self.native_size = self.base.get_size()

    def set_emotion(self, e): self.emotion = e
    def set_mouth(self, m): self.mouth = m

    def draw_scaled(self, surf, rect: pygame.Rect):
        pad = 12; aw, ah = rect.width - 2 * pad, rect.height - 2 * pad; bw, bh = self.native_size
        s = min(aw / bw, ah / bh); nw, nh = int(bw * s), int(bh * s)
        x = rect.x + (rect.width - nw) // 2; y = rect.y + (rect.height - nh) // 2
        base = pygame.transform.smoothscale(self.base, (nw, nh))
        eyes = pygame.transform.smoothscale(self.eyes[self.emotion], (nw, nh))
        mouth_key = self.mouth
        if self.emotion == 'Happy' and self.mouth in ('closed', 'smile'): mouth_key = 'smile'
        if self.emotion == 'Surprised' and self.mouth in ('open', 'o', 'teeth'): mouth_key = 'surprised'
        mouth = pygame.transform.smoothscale(self.mouths[mouth_key], (nw, nh))
        surf.blit(base, (x, y)); surf.blit(eyes, (x, y)); surf.blit(mouth, (x, y))


# ----------------------------- Prehrávač s lip-sync -----------------------------
class SpeechPlayer:
    def __init__(self):
        self.playing = False; self.start_t = 0.0; self.rms = []; self.rms_s = 0.0; self.wav = None; self.duration = 0.0
        pygame.mixer.init(frequency=22050, channels=2)

    def load_and_play(self, wav_path: str, volume: float = 0.9):
        self.stop(); self.wav = wav_path
        self.rms, _, self.duration = wav_rms_timeline(wav_path)
        self.rms_s = 0.0; pygame.mixer.music.load(wav_path); pygame.mixer.music.set_volume(volume); pygame.mixer.music.play()
        self.start_t = time.time(); self.playing = True

    def stop(self):
        if self.playing: pygame.mixer.music.stop()
        self.playing = False; self.rms = []
        if self.wav and os.path.exists(self.wav):
            try: os.remove(self.wav)
            except Exception: pass
        self.wav = None

    def progress(self) -> float:
        if not self.playing or self.duration <= 0: return 0.0
        p = (time.time() - self.start_t) / self.duration
        return max(0.0, min(1.0, p))

    def current_rms(self) -> float:
        if not self.playing or not self.rms: return 0.0
        idx = int(((time.time() - self.start_t) * 1000.0) // FRAME_MS)
        if idx >= len(self.rms): self.stop(); return 0.0
        inst = self.rms[idx]; self.rms_s = RMS_SMOOTH * self.rms_s + (1.0 - RMS_SMOOTH) * inst
        return self.rms_s


# ----------------------------- App -----------------------------
class App:
    def __init__(self):
        pygame.init()
        pygame.key.set_repeat(400, 30)  # pre držanie backspace
        pygame.display.set_caption('Hovoriaci avatar – offline')
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.font = pygame.font.Font(FONT_NAME, 20)
        self.font_s = pygame.font.Font(FONT_NAME, 14)
        self.font_t = pygame.font.Font(FONT_NAME, 24)
        self.avatar = Avatar()
        self.player = SpeechPlayer()

        rx = LEFT_W + PAD
        y = PAD + 48
        self.box = TextBox((rx, y, RIGHT_W - 2 * PAD, CTRL_H))
        y += CTRL_H + GAP
        half = (RIGHT_W - 3 * PAD) // 2
        self.btn_speak = Button((rx, y, half, CTRL_H), 'Speak', self.on_speak)
        self.btn_stop = Button((rx + half + PAD, y, half, CTRL_H), 'Stop', self.on_stop)
        y += CTRL_H + GAP
        self.tgl_lang = Dropdown(pygame.Rect(rx, y, RIGHT_W - 2 * PAD, CTRL_H + SMALL_H), ['SK', 'EN'], 0, 'Jazyk')
        y += CTRL_H + SMALL_H + GAP
        self.emotion_windows: List[Tuple[float, float, str]] = []
        self.base_emo = 'Neutral'

    def on_speak(self):
        text = self.box.text.strip()
        if not text: return
        wav = espeak_wav(text, self.tgl_lang.value, ESPEAK_WPM)
        self.emotion_windows = find_emotion_windows(text)
        self.base_emo = base_emotion(text)
        self.player.load_and_play(wav, volume=0.9)

    def on_stop(self):
        self.player.stop()

    def draw_meter(self, rms: float):
        v = min(rms / RMS_METER_CLAMP, 1.0)
        h = 80; w = 10
        x = LEFT_W + RIGHT_W - PAD - w - 6
        y = PAD + 48
        pygame.draw.rect(self.screen, (55, 58, 66), (x, y, w, h), border_radius=4)
        pygame.draw.rect(self.screen, ACC, (x, y + h - int(h * v), w, int(h * v)), border_radius=4)

    def current_emotion(self) -> str:
        p = self.player.progress()
        for a, b, e in self.emotion_windows:
            if a <= p <= b: return e
        return self.base_emo

    def run(self):
        clock = pygame.time.Clock()
        running = True
        LAYOUT_W = 400
        LAYOUT_X = (WINDOW_W - LAYOUT_W) // 2

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN and self.box.active:
                    self.on_speak()
                self.box.handle(event, self.font)
                self.btn_speak.handle(event)
                self.btn_stop.handle(event)
                self.tgl_lang.handle(event)

            self.screen.fill(BG)

            # Avatar
            avatar_h = 420
            avatar_rect = pygame.Rect(LAYOUT_X, 40, LAYOUT_W, avatar_h)
            pygame.draw.rect(self.screen, CARD, avatar_rect, border_radius=12)
            pygame.draw.rect(self.screen, (70, 70, 80), avatar_rect, width=2, border_radius=12)

            rms = self.player.current_rms()
            state = "closed"
            if rms >= RMS_THR4: state = "o"
            elif rms >= RMS_THR3: state = "open"
            elif rms >= RMS_THR2: state = "teeth"
            elif rms >= RMS_THR1: state = "smile" if self.current_emotion() == "Happy" else "teeth"

            emo = self.current_emotion()
            self.avatar.set_emotion(emo)
            self.avatar.set_mouth(state)
            inner = avatar_rect.inflate(-16, -16)
            self.avatar.draw_scaled(self.screen, inner)

            # UI
            spacing = 20
            current_y = avatar_rect.bottom + spacing

            self.box.rect = pygame.Rect(LAYOUT_X, current_y, LAYOUT_W, 44)
            self.box.draw(self.screen, self.font)
            current_y += 44 + spacing

            badge = pygame.Rect(LAYOUT_X, current_y, LAYOUT_W, 26)
            pygame.draw.rect(self.screen, CARD, badge, border_radius=8)
            pygame.draw.rect(self.screen, (70, 70, 80), badge, width=1, border_radius=8)
            text = self.font_s.render(f"Emócia: {emo}", True, TXT)
            self.screen.blit(text, (badge.centerx - text.get_width() // 2, badge.y + 4))
            current_y += badge.height + spacing

            btn_h = 44; btn_gap = 12; btn_w = (LAYOUT_W - btn_gap) // 2
            self.btn_speak.rect = pygame.Rect(LAYOUT_X, current_y, btn_w, btn_h)
            self.btn_stop.rect = pygame.Rect(LAYOUT_X + btn_w + btn_gap, current_y, btn_w, btn_h)
            self.btn_speak.draw(self.screen, self.font)
            self.btn_stop.draw(self.screen, self.font)
            current_y += btn_h + spacing

            toggle_h = 50
            self.tgl_lang.rect = pygame.Rect(LAYOUT_X, current_y, LAYOUT_W, toggle_h)
            self.tgl_lang.draw(self.screen, self.font, self.font_s)

            self.draw_meter(rms)
            pygame.display.flip()
            clock.tick(60)

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
