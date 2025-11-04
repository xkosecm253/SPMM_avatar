"""
Hovoriaci avatar – OFFLINE (pygame + eSpeak‑NG)
------------------------------------------------
Stabilná verzia bez zbytočných ovládačov, s:
• dvojpanelovým UI (vľavo avatar, vpravo ovládanie),
• lip‑syncom cez RMS + vyhladzovanie a 5 tvarov úst (closed/smile/teeth/open/o),
• AUTO emóciami z textu (lexikálny slovník SK/EN),
• „emočné okná“ – ak sa v texte vyskytne pozitívne/slovo prekvapenia, avatar prehodí výraz
  PRIBLIŽNE v tom úseku audia (podľa pozície slova v texte a dĺžky WAVu),
• bezpečné ovládanie: Enter spúšťa reč iba keď je aktívne textové pole; SPACE nič nespúšťa,
  aby to nespadlo pri písaní.

Požiadavky:  pip install pygame numpy   +   nainštalovaný eSpeak‑NG
Assets (./assets):
  base_face.png
  natural_eyes.png, happy_eyes.png, surprised_eyes.png
  closed_mouth.png, open_mouth.png, happy_mouth.png, surprised_mouth.png
  o_mouth.png, smile_mouth.png, teeth_mouth.png

Spustenie:  python avatar.py
"""

import os, subprocess, tempfile, wave, time, re
from typing import List, Tuple
import numpy as np
import pygame

# ----------------------------- Konštanty -----------------------------
ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')


#WINDOW_W, WINDOW_H = 1100, 720 
#stare nastavenia okna

WINDOW_W = 480
WINDOW_H = 740

BG = (16,18,22); PANEL = (26,28,34); CARD = (32,35,42); TXT=(230,230,235); MUTED=(165,170,180); ACC=(120,180,255)
FONT_NAME = 'freesansbold.ttf'

# Lip‑sync – rámec a prahy
FRAME_MS = 20
RMS_THR1 = 0.018  # closed→smile/teeth
RMS_THR2 = 0.030  # → teeth
RMS_THR3 = 0.045  # → open
RMS_THR4 = 0.070  # → o
RMS_SMOOTH = 0.6  # exponenciálne vyhladzovanie
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
    cmd = ['espeak-ng','-v',voice,'-s',str(wpm),'-w',wav_path,text]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError('espeak-ng nie je v PATH. Doinštaluj eSpeak NG.')
    return wav_path


def wav_rms_timeline(path: str, frame_ms: int=FRAME_MS) -> Tuple[List[float], int, float]:
    with wave.open(path,'rb') as wf:
        ch = wf.getnchannels(); sw = wf.getsampwidth(); sr = wf.getframerate(); n = wf.getnframes()
        raw = wf.readframes(n)
    if sw==2:
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)/32768.0
    elif sw==1:
        a = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32)-128)/128.0
    else:
        raise ValueError('Nepodporovaný WAV')
    if ch>1:
        a = a.reshape(-1,ch).mean(axis=1)
    frame_len = int(sr*frame_ms/1000)
    rms = [float(np.sqrt(np.mean(np.square(a[i:i+frame_len])))) for i in range(0,len(a),frame_len)]
    duration = len(a)/sr
    return rms, sr, duration

# ----------------------------- Text → Emócia -----------------------------
POS_WORDS = [
    'teším sa','tesim sa','veľmi rád','velmi rad','super','skvel','paráda','parada','radosť','mam rad','mám rád','milujem','great','awesome','happy','glad','love'
]
SURP_WORDS = ['čože','coze','wow','neverím','neverim','vážne','vazne','fíha','fiha','prekvap','?!','!?']

_word_re = re.compile(r"[\wáäčďéíľĺňóôŕřšťúýž]+", re.IGNORECASE)

def find_emotion_windows(text: str) -> List[Tuple[float,float,str]]:
    """Vytvorí zoznam okien v percentách trvania audia.
    Každé okno = (start_pct, end_pct, emotion). Jednoduchý odhad podľa pozície slova v texte.
    """
    t = text.lower()
    tokens = _word_re.findall(t)
    if not tokens:
        return []
    total = len(tokens)
    windows: List[Tuple[float,float,str]] = []
    for i, w in enumerate(tokens):
        pct = i/total
        if any(k.replace(' ','') in w.replace(' ','') for k in POS_WORDS):
            windows.append((max(0.0,pct-0.05), min(1.0,pct+0.05), 'Happy'))
        if any(k.replace(' ','') in w.replace(' ','') for k in SURP_WORDS):
            windows.append((max(0.0,pct-0.05), min(1.0,pct+0.05), 'Surprised'))
    return windows


def base_emotion(text: str) -> str:
    t = text.lower()
    if any(k.replace(' ','') in t.replace(' ','') for k in SURP_WORDS):
        return 'Surprised'
    if any(k.replace(' ','') in t.replace(' ','') for k in POS_WORDS):
        return 'Happy'
    return 'Neutral'

# ----------------------------- UI prvky -----------------------------
class Button:
    def __init__(self, rect, label, on_click):
        self.rect = pygame.Rect(rect); self.label=label; self.on_click=on_click
    def draw(self,surf,font):
        pygame.draw.rect(surf, CARD, self.rect, border_radius=8)
        pygame.draw.rect(surf, ACC, self.rect, width=2, border_radius=8)
        txt = font.render(self.label, True, TXT)
        surf.blit(txt, txt.get_rect(center=self.rect.center))
    def handle(self,event):
        if event.type==pygame.MOUSEBUTTONDOWN and event.button==1 and self.rect.collidepoint(event.pos):
            self.on_click()

class Toggle:
    def __init__(self, rect, options, idx=0, label=''):
        self.rect=pygame.Rect(rect); self.options=options; self.idx=idx; self.label=label
    @property
    def value(self): return self.options[self.idx]
    def next(self): self.idx=(self.idx+1)%len(self.options)
    def draw(self,surf,font,font_small):
        lab = font_small.render(self.label, True, MUTED)
        surf.blit(lab,(self.rect.x+2,self.rect.y))
        box = pygame.Rect(self.rect.x, self.rect.y+SMALL_H, self.rect.width, CTRL_H)
        pygame.draw.rect(surf, CARD, box, border_radius=8)
        pygame.draw.rect(surf, (70,70,80), box, width=2, border_radius=8)
        surf.blit(font.render(self.value, True, TXT),(box.x+12, box.y+10))
    def handle(self,event):
        box = pygame.Rect(self.rect.x, self.rect.y+SMALL_H, self.rect.width, CTRL_H)
        if event.type==pygame.MOUSEBUTTONDOWN and event.button==1 and box.collidepoint(event.pos):
            self.next()

class TextBox:
    def __init__(self, rect, text=''):
        self.rect=pygame.Rect(rect); self.text=text; self.active=False
    def draw(self,surf,font,placeholder='Napíš text…'):
        pygame.draw.rect(surf, CARD, self.rect, border_radius=8)
        pygame.draw.rect(surf, ACC if self.active else (70,70,80), self.rect, width=2, border_radius=8)
        shown=self.text if self.text else placeholder
        color=TXT if self.text else MUTED
        clip=pygame.Rect(self.rect.x+10,self.rect.y+10,self.rect.width-20,self.rect.height-20)
        surf.set_clip(clip); surf.blit(font.render(shown,True,color),(self.rect.x+10,self.rect.y+10)); surf.set_clip(None)
    def handle(self,event):
        if event.type==pygame.MOUSEBUTTONDOWN: self.active=self.rect.collidepoint(event.pos)
        if self.active and event.type==pygame.KEYDOWN:
            if event.key==pygame.K_BACKSPACE: self.text=self.text[:-1]
            elif event.key==pygame.K_RETURN:  pass
            else:
                if event.unicode: self.text+=event.unicode

# ----------------------------- Avatar -----------------------------
class Avatar:
    def __init__(self):
        self.base=load_img('base_face.png')
        self.eyes={'Neutral':load_img('natural_eyes.png'),'Happy':load_img('happy_eyes.png'),'Surprised':load_img('surprised_eyes.png')}
        self.mouths={
            'closed':load_img('closed_mouth.png'),
            'smile': load_img('smile_mouth.png'),
            'teeth': load_img('teeth_mouth.png'),
            'open':  load_img('open_mouth.png'),
            'o':     load_img('o_mouth.png'),
            'happy': load_img('happy_mouth.png'),
            'surprised':load_img('surprised_mouth.png')}
        self.emotion='Neutral'; self.mouth='closed'; self.native_size=self.base.get_size()
    def set_emotion(self,e): self.emotion=e
    def set_mouth(self,m): self.mouth=m
    def draw_scaled(self,surf,rect:pygame.Rect):
        pad=12; aw,ah=rect.width-2*pad, rect.height-2*pad; bw,bh=self.native_size
        s=min(aw/bw, ah/bh); nw,nh=int(bw*s), int(bh*s); x=rect.x+(rect.width-nw)//2; y=rect.y+(rect.height-nh)//2
        base=pygame.transform.smoothscale(self.base,(nw,nh)); eyes=pygame.transform.smoothscale(self.eyes[self.emotion],(nw,nh))
        mouth_key=self.mouth
        if self.emotion=='Happy' and self.mouth in ('closed','smile'): mouth_key='smile'
        if self.emotion=='Surprised' and self.mouth in ('open','o','teeth'): mouth_key='surprised'
        mouth=pygame.transform.smoothscale(self.mouths[mouth_key],(nw,nh))
        surf.blit(base,(x,y)); surf.blit(eyes,(x,y)); surf.blit(mouth,(x,y))

# ----------------------------- Prehrávač s lip‑sync -----------------------------
class SpeechPlayer:
    def __init__(self):
        self.playing=False; self.start_t=0.0; self.rms=[]; self.rms_s=0.0; self.wav=None; self.duration=0.0
        pygame.mixer.init(frequency=22050, channels=2)
    def load_and_play(self,wav_path:str, volume:float=0.9):
        self.stop(); self.wav=wav_path
        self.rms, _, self.duration = wav_rms_timeline(wav_path)
        self.rms_s=0.0; pygame.mixer.music.load(wav_path); pygame.mixer.music.set_volume(volume); pygame.mixer.music.play()
        self.start_t=time.time(); self.playing=True
    def stop(self):
        if self.playing: pygame.mixer.music.stop()
        self.playing=False; self.rms=[]
        if self.wav and os.path.exists(self.wav):
            try: os.remove(self.wav)
            except Exception: pass
        self.wav=None
    def progress(self)->float:
        if not self.playing or self.duration<=0: return 0.0
        p=(time.time()-self.start_t)/self.duration
        return max(0.0, min(1.0, p))
    def current_rms(self)->float:
        if not self.playing or not self.rms: return 0.0
        idx=int(((time.time()-self.start_t)*1000.0)//FRAME_MS)
        if idx>=len(self.rms): self.stop(); return 0.0
        inst=self.rms[idx]; self.rms_s=RMS_SMOOTH*self.rms_s+(1.0-RMS_SMOOTH)*inst
        return self.rms_s

# ----------------------------- App -----------------------------
class App:
    def __init__(self):
        pygame.init(); pygame.display.set_caption('Hovoriaci avatar – offline')
        self.screen=pygame.display.set_mode((WINDOW_W,WINDOW_H))
        self.font=pygame.font.Font(FONT_NAME,20); self.font_s=pygame.font.Font(FONT_NAME,14); self.font_t=pygame.font.Font(FONT_NAME,24)
        self.avatar=Avatar(); self.player=SpeechPlayer()
        rx=LEFT_W+PAD; y=PAD+48
        self.box=TextBox((rx,y,RIGHT_W-2*PAD,CTRL_H)); y+=CTRL_H+GAP
        half=(RIGHT_W-3*PAD)//2
        self.btn_speak=Button((rx,y,half,CTRL_H),'▶ Speak', self.on_speak)
        self.btn_stop =Button((rx+half+PAD,y,half,CTRL_H),'■ Stop', self.on_stop); y+=CTRL_H+GAP
        self.tgl_lang=Toggle((rx,y,RIGHT_W-2*PAD,CTRL_H+SMALL_H),['SK','EN'],0,'Jazyk'); y+=CTRL_H+SMALL_H+GAP
        self.emotion_windows: List[Tuple[float,float,str]] = []
        self.base_emo = 'Neutral'

    def on_speak(self):
        text=self.box.text.strip()
        if not text: return
        wav=espeak_wav(text, self.tgl_lang.value, ESPEAK_WPM)
        self.emotion_windows = find_emotion_windows(text)
        self.base_emo = base_emotion(text)
        self.player.load_and_play(wav, volume=0.9)

    def on_stop(self):
        self.player.stop()

    def draw_meter(self, rms: float):
        v=min(rms/RMS_METER_CLAMP,1.0); h=80; w=10; x=LEFT_W+RIGHT_W-PAD-w-6; y=PAD+48
        pygame.draw.rect(self.screen,(55,58,66),(x,y,w,h),border_radius=4)
        pygame.draw.rect(self.screen,ACC,(x,y+h-int(h*v),w,int(h*v)),border_radius=4)

    def draw_layout(self):
        self.screen.fill(BG)
        left=pygame.Rect(0,0,LEFT_W,WINDOW_H); right=pygame.Rect(LEFT_W,0,RIGHT_W,WINDOW_H)
        pygame.draw.rect(self.screen,(26,28,34),left); pygame.draw.rect(self.screen,PANEL,right)
        self.screen.blit(self.font_t.render('Hovoriaci avatar – offline',True,TXT),(LEFT_W+PAD,PAD))
        self.screen.blit(self.font_s.render('pygame + eSpeak NG',True,MUTED),(LEFT_W+PAD,PAD+22))
        pygame.draw.line(self.screen,(50,50,58),(LEFT_W,0),(LEFT_W,WINDOW_H),1)
        return left,right

    def current_emotion(self)->str:
        p=self.player.progress()
        for (a,b,e) in self.emotion_windows:
            if a<=p<=b: return e
        return self.base_emo

    """""
    def run(self):
        clock=pygame.time.Clock(); running=True
        while running:
            for event in pygame.event.get():
                if event.type==pygame.QUIT: running=False
                if event.type==pygame.KEYDOWN and event.key==pygame.K_RETURN and self.box.active:
                    self.on_speak()
                self.box.handle(event); self.btn_speak.handle(event); self.btn_stop.handle(event); self.tgl_lang.handle(event)

            self.draw_layout()
            rms=self.player.current_rms()
            r=rms
            if r<RMS_THR1: state='closed'
            elif r<RMS_THR2: state='teeth'
            elif r<RMS_THR3: state='open'
            else: state='o'
            emo=self.current_emotion(); self.avatar.set_emotion(emo)
            if emo=='Happy' and r<RMS_THR2: state='smile'
            self.avatar.set_mouth(state)

            area=pygame.Rect(PAD,PAD,LEFT_W-2*PAD,WINDOW_H-2*PAD)
            self.avatar.draw_scaled(self.screen, area)

            self.box.draw(self.screen,self.font)
            self.btn_speak.draw(self.screen,self.font)
            self.btn_stop.draw(self.screen,self.font)
            self.tgl_lang.draw(self.screen,self.font,self.font_s)
            badge=pygame.Rect(LEFT_W+PAD, PAD+82, 200, 24)
            pygame.draw.rect(self.screen,CARD,badge,border_radius=6); pygame.draw.rect(self.screen,(70,70,80),badge,width=1,border_radius=6)
            self.screen.blit(self.font_s.render(f'Emócia: {emo}',True,TXT),(badge.x+8,badge.y+4))

            self.draw_meter(rms)
            pygame.display.flip(); clock.tick(60)
        self.player.stop(); pygame.quit()
    """""


    def run(self):
        clock = pygame.time.Clock()
        running = True

        # fixná šírka layoutu (podľa textboxu)
        LAYOUT_W = 400
        LAYOUT_X = (WINDOW_W - LAYOUT_W) // 2

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN and self.box.active:
                    self.on_speak()
                self.box.handle(event)
                self.btn_speak.handle(event)
                self.btn_stop.handle(event)
                self.tgl_lang.handle(event)

            # === Pozadie ===
            self.screen.fill(BG)

            # === Avatar ===
            avatar_h = 420
            avatar_rect = pygame.Rect(LAYOUT_X, 40, LAYOUT_W, avatar_h)
            pygame.draw.rect(self.screen, CARD, avatar_rect, border_radius=12)
            pygame.draw.rect(self.screen, (70, 70, 80), avatar_rect, width=2, border_radius=12)

            rms = self.player.current_rms()
            if rms < RMS_THR1:
                state = "closed"
            elif rms < RMS_THR2:
                state = "teeth"
            elif rms < RMS_THR3:
                state = "open"
            else:
                state = "o"
            emo = self.current_emotion()
            self.avatar.set_emotion(emo)
            if emo == "Happy" and rms < RMS_THR2:
                state = "smile"
            self.avatar.set_mouth(state)

            # vykresli avatar
            inner = avatar_rect.inflate(-16, -16)
            self.avatar.draw_scaled(self.screen, inner)

            # === UI prvky ===
            spacing = 20
            current_y = avatar_rect.bottom + spacing

            # Textové pole
            self.box.rect = pygame.Rect(LAYOUT_X, current_y, LAYOUT_W, 44)
            self.box.draw(self.screen, self.font)
            current_y += 44 + spacing

            # Emócia (malý badge)
            badge = pygame.Rect(LAYOUT_X, current_y, LAYOUT_W, 26)
            pygame.draw.rect(self.screen, CARD, badge, border_radius=8)
            pygame.draw.rect(self.screen, (70, 70, 80), badge, width=1, border_radius=8)
            text = self.font_s.render(f"Emócia: {emo}", True, TXT)
            text_x = badge.centerx - text.get_width() // 2
            self.screen.blit(text, (text_x, badge.y + 4))
            current_y += badge.height + spacing

            # Tlačidlá (vedľa seba, ale celková šírka = textbox)
            btn_h = 44
            btn_gap = 12
            btn_w = (LAYOUT_W - btn_gap) // 2
            self.btn_speak.rect = pygame.Rect(LAYOUT_X, current_y, btn_w, btn_h)
            self.btn_stop.rect = pygame.Rect(LAYOUT_X + btn_w + btn_gap, current_y, btn_w, btn_h)
            self.btn_speak.draw(self.screen, self.font)
            self.btn_stop.draw(self.screen, self.font)
            current_y += btn_h + spacing

            # Prepínač jazyka
            toggle_h = 50
            self.tgl_lang.rect = pygame.Rect(LAYOUT_X, current_y, LAYOUT_W, toggle_h)
            self.tgl_lang.draw(self.screen, self.font, self.font_s)

            # === Zvukový meter ===
            self.draw_meter(rms)

            pygame.display.flip()
            clock.tick(60)
        self.player.stop()
        pygame.quit()


if __name__=='__main__':
    try:
        App().run()
    except FileNotFoundError as e:
        print('\n[CHÝBA ASSET]', e); print('Skontroluj ./assets názvy súborov.')
    except Exception as ex:
        print('Chyba:', ex)
