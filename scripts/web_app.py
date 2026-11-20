"""EdgeBot web control panel: drive the robot from a browser.

    python scripts/web_app.py            # then open http://localhost:8000

Type or speak a command, toggle the lidar safety shield, watch the robot's run
as a video, hear its spoken reply, and run the shield experiment live. Uses only
the Python standard library for the server; everything runs locally.
"""

import argparse
import json
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_demo import load_stack  # noqa: E402

from edgebot.robot.agent import EdgeBot, FrameSink  # noqa: E402

MEDIA = Path(tempfile.mkdtemp(prefix="edgebot_web_"))
LOCK = threading.Lock()
STATE: dict = {}


def new_bot():
    sink = FrameSink(enabled=True)
    bot = EdgeBot(STATE["nlu"], STATE["policy"], STATE["asr"], STATE["tts"], sink)
    sink.compose()
    return bot


def run_turn(text=None, audio=None, shield=True):
    bot = STATE["bot"]
    bot.use_shield = shield
    start = len(bot.sink.frames)
    before = dict(bot.stats)
    turn = bot.hear(audio) if audio is not None else bot.handle(text)
    frames = bot.sink.frames[start:]
    stamp = f"{int(time.time() * 1000)}"
    video = None
    if frames:
        import imageio.v2 as imageio

        video = f"run_{stamp}.mp4"
        with imageio.get_writer(MEDIA / video, fps=20, codec="libx264", quality=7, macro_block_size=4) as w:
            for f in frames:
                w.append_data(f)
    bot.sink.frames = bot.sink.frames[-1:]  # keep memory bounded
    reply = None
    if STATE["tts"] is not None:
        reply = f"reply_{stamp}.wav"
        bot.speak(turn.response, MEDIA / reply)
    delta = {k: bot.stats[k] - before[k] for k in bot.stats}
    return {"user": turn.user, "heard": turn.asr_text, "frame": turn.frame, "response": turn.response,
            "success": turn.success, "timings": turn.timings, "stats": bot.stats, "delta": delta,
            "battery": round(bot.battery, 1), "distance_m": round(bot.distance_travelled, 2),
            "station": bot.station, "holding": bot.holding, "shield": shield,
            "video": video and f"/media/{video}", "reply": reply and f"/media/{reply}"}


def shield_test(episodes):
    from edgebot.edge.export import OnnxPolicy
    from edgebot.rl.evaluate import evaluate_nav
    from edgebot.utils import CHECKPOINTS

    rows = []
    for name, f in (("PPO FP32", "nav_ppo_actor.onnx"), ("PPO INT8", "nav_ppo_actor_int8.onnx")):
        pol = OnnxPolicy(CHECKPOINTS / "onnx" / f)
        for sh in (False, True):
            m = evaluate_nav(pol, episodes, shield=sh)
            rows.append({"model": name, "shield": sh, "success": m["success_rate"],
                         "collision": m["collision_rate"], "timeout": m["timeout_rate"],
                         "active": m.get("shield_intervention_rate", 0.0)})
    return rows


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/media/"):
            f = MEDIA / Path(self.path).name
            if not f.exists():
                return self._json({"error": "not found"}, 404)
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4" if f.suffix == ".mp4" else "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/api/info":
            self._json({"runtime": STATE["runtime"], "asr": STATE["asr"] is not None,
                        "tts": STATE["tts"] is not None})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        try:
            with LOCK:
                if self.path == "/api/command":
                    req = json.loads(raw)
                    return self._json(run_turn(text=req["text"], shield=req.get("shield", True)))
                if self.path.startswith("/api/voice"):
                    if STATE["asr"] is None:
                        return self._json({"error": "speech recognition is off (started with --no-speech)"}, 400)
                    from faster_whisper import decode_audio

                    tmp = MEDIA / f"in_{int(time.time() * 1000)}.webm"
                    tmp.write_bytes(raw)
                    audio = decode_audio(str(tmp), sampling_rate=16_000)
                    return self._json(run_turn(audio=audio, shield="shield=0" not in self.path))
                if self.path == "/api/reset":
                    STATE["bot"] = new_bot()
                    return self._json({"ok": True})
                if self.path == "/api/shield_test":
                    eps = int(json.loads(raw or b"{}").get("episodes", 30))
                    return self._json({"episodes": eps, "rows": shield_test(eps)})
            self._json({"error": "not found"}, 404)
        except Exception as e:  # show the error in the page instead of dropping the connection
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>EdgeBot Control Panel</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1b1f24;--mute:#5d6672;--line:#dde1e6;--acc:#2563eb;--ok:#15803d;--bad:#b91c1c;--warn:#c2410c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,sans-serif}
header{padding:14px 20px;background:var(--card);border-bottom:1px solid var(--line)}
header h1{margin:0;font-size:19px}header p{margin:2px 0 0;color:var(--mute);font-size:13px}
main{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(300px,1fr);gap:16px;padding:16px 20px;max-width:1400px;margin:auto}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
h2{font-size:14px;margin:0 0 10px;text-transform:uppercase;letter-spacing:.04em;color:var(--mute)}
video{width:100%;border-radius:6px;background:#eee;aspect-ratio:1140/640}
.row{display:flex;gap:8px;flex-wrap:wrap}input[type=text]{flex:1;min-width:200px;padding:10px;border:1px solid var(--line);border-radius:6px;font:inherit}
button{padding:9px 14px;border:1px solid var(--line);background:var(--card);border-radius:6px;font:inherit;cursor:pointer}
button.pri{background:var(--acc);color:#fff;border-color:var(--acc)}button:disabled{opacity:.5;cursor:wait}
button.rec{background:var(--bad);color:#fff;border-color:var(--bad)}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}.chips button{font-size:13px;padding:5px 9px}
.pipe{display:grid;grid-template-columns:120px 1fr;gap:4px 10px;font-size:14px}.pipe b{color:var(--mute);font-weight:600}
code{background:#f0f2f5;padding:1px 5px;border-radius:4px;font-size:13px}
.stat{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.stat div{background:var(--bg);border-radius:6px;padding:8px}
.stat span{display:block;font-size:12px;color:var(--mute)}.stat strong{font-size:18px}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:5px;border-bottom:1px solid var(--line);text-align:right}td:first-child,th:first-child{text-align:left}
.log{max-height:240px;overflow:auto;font-size:13px}.log p{margin:4px 0}.u{color:var(--acc)}.b{color:var(--ok)}
.switch{display:flex;align-items:center;gap:8px;margin-top:10px}.err{color:var(--bad)}.muted{color:var(--mute);font-size:13px}
</style></head><body>
<header><h1>EdgeBot Control Panel</h1><p>Voice-commanded warehouse robot &middot; PPO navigation + lidar safety shield &middot; running locally on this laptop</p></header>
<main>
<section>
 <div class="card"><h2>Robot view</h2>
  <video id="vid" controls muted playsinline></video>
  <p class="muted">Left: warehouse map (purple = A* route, blue = driven path, red rays = 24-beam lidar). Right: onboard arm.</p>
 </div>
 <div class="card" style="margin-top:16px"><h2>Command the robot</h2>
  <div class="row"><input id="txt" type="text" placeholder="e.g. move the red box from shelf a to the conveyor">
   <button class="pri" id="send">Send</button><button id="mic">🎤 Speak</button></div>
  <div class="chips" id="chips"></div>
  <label class="switch"><input type="checkbox" id="shield" checked> Lidar safety shield ON (my novelty)</label>
  <div class="row" style="margin-top:10px"><button id="reset">Reset warehouse</button><span id="busy" class="muted"></span></div>
 </div>
</section>
<section>
 <div class="card"><h2>What happened (pipeline)</h2><div class="pipe" id="pipe"><b>Status</b><span class="muted">Send a command to start.</span></div></div>
 <div class="card" style="margin-top:16px"><h2>Robot state</h2><div class="stat" id="stat"></div></div>
 <div class="card" style="margin-top:16px"><h2>Novelty experiment: shield off vs on</h2>
  <div class="row"><button id="exp">Run on 30 unseen warehouses (~1 min)</button></div><div id="expout" style="margin-top:10px"></div></div>
 <div class="card" style="margin-top:16px"><h2>Conversation</h2><div class="log" id="log"></div></div>
</section></main>
<script>
const $=id=>document.getElementById(id);
const EX=["hello jetbot","move the red box from shelf a to the conveyor","pick up the box","the blue one","put it on the packing station","stack the red green and blue blocks at the workbench","what is your status","go back to the charging dock"];
EX.forEach(t=>{const b=document.createElement('button');b.textContent=t;b.onclick=()=>{$('txt').value=t;send()};$('chips').appendChild(b)});
function busy(on,msg){document.querySelectorAll('button').forEach(b=>b.disabled=on);$('busy').textContent=on?msg:''}
const pct=x=>(100*x).toFixed(1)+'%';
function esc(s){return String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function render(r){
 if(r.error){$('pipe').innerHTML='<b>Error</b><span class="err">'+esc(r.error)+'</span>';return}
 const slots=Object.entries(r.frame.slots||{}).map(([k,v])=>'<code>'+esc(k)+'='+esc(v)+'</code>').join(' ')||'<span class="muted">none</span>';
 const d=r.delta;
 $('pipe').innerHTML=
  (r.heard!=null?'<b>1. Whisper heard</b><span>"'+esc(r.heard)+'"</span>':'<b>1. Input</b><span>"'+esc(r.user)+'" (typed)</span>')+
  '<b>2. NLU intent</b><span><code>'+esc(r.frame.intent)+'</code> (confidence '+(r.frame.confidence*100).toFixed(1)+'%)</span>'+
  '<b>3. NLU slots</b><span>'+slots+'</span>'+
  '<b>4. PPO driving</b><span>'+(d.nav_attempts?d.nav_success+'/'+d.nav_attempts+' drives reached the goal, '+d.nav_steps+' control steps':'no driving needed')+'</span>'+
  '<b>5. Collisions</b><span style="color:'+(d.collisions?'var(--bad)':'var(--ok)')+'">'+d.collisions+' (shield '+(r.shield?'ON':'OFF')+')</span>'+
  '<b>6. Arm</b><span>'+(d.picks||d.places?d.picks+' pick, '+d.places+' place':'not used')+'</span>'+
  '<b>7. Reply</b><span>'+esc(r.response)+'</span>'+
  '<b>Latency</b><span>NLU '+r.timings.nlu_ms.toFixed(1)+' ms'+(r.timings.asr_s!=null?', ASR '+r.timings.asr_s.toFixed(2)+' s':'')+'</span>';
 const s=r.stats;
 $('stat').innerHTML=[['Location',r.station],['Holding',r.holding?r.holding.join(' '):'nothing'],['Battery',r.battery+'%'],
  ['Distance',r.distance_m+' m'],['Drives OK',s.nav_success+'/'+s.nav_attempts],['Collisions',s.collisions]]
  .map(([k,v])=>'<div><span>'+k+'</span><strong>'+esc(v)+'</strong></div>').join('');
 $('log').insertAdjacentHTML('beforeend','<p class="u">You: '+esc(r.heard??r.user)+'</p><p class="b">EdgeBot: '+esc(r.response)+'</p>');
 $('log').scrollTop=1e9;
 if(r.video){$('vid').src=r.video;$('vid').play().catch(()=>{})}
 if(r.reply){new Audio(r.reply).play().catch(()=>{})}
}
async function post(url,body,type){const r=await fetch(url,{method:'POST',headers:{'Content-Type':type||'application/json'},body});return r.json()}
async function send(){const t=$('txt').value.trim();if(!t)return;busy(true,'Robot working…');
 try{render(await post('/api/command',JSON.stringify({text:t,shield:$('shield').checked})))}catch(e){render({error:e})}busy(false)}
$('send').onclick=send;$('txt').onkeydown=e=>{if(e.key==='Enter')send()};
$('reset').onclick=async()=>{busy(true,'Resetting…');await post('/api/reset','{}');$('log').innerHTML='';$('pipe').innerHTML='<b>Status</b><span>Warehouse reset.</span>';busy(false)};
let rec=null,chunks=[];
$('mic').onclick=async()=>{
 if(rec){rec.stop();return}
 try{const st=await navigator.mediaDevices.getUserMedia({audio:true});rec=new MediaRecorder(st);chunks=[];
  rec.ondataavailable=e=>chunks.push(e.data);
  rec.onstop=async()=>{st.getTracks().forEach(t=>t.stop());const blob=new Blob(chunks,{type:rec.mimeType});rec=null;
   $('mic').className='';$('mic').textContent='🎤 Speak';busy(true,'Transcribing and acting…');
   try{render(await post('/api/voice?shield='+($('shield').checked?1:0),blob,'application/octet-stream'))}catch(e){render({error:e})}busy(false)};
  rec.start();$('mic').className='rec';$('mic').textContent='■ Stop';$('mic').disabled=false}
 catch(e){render({error:'Microphone not available: '+e})}};
$('exp').onclick=async()=>{busy(true,'Running 120 episodes…');$('expout').innerHTML='<span class="muted">Running…</span>';
 const r=await post('/api/shield_test',JSON.stringify({episodes:30}));
 if(r.error){$('expout').innerHTML='<span class="err">'+esc(r.error)+'</span>';busy(false);return}
 $('expout').innerHTML='<table><tr><th>Policy</th><th>Shield</th><th>Success</th><th>Collision</th><th>Timeout</th><th>Active</th></tr>'+
  r.rows.map(x=>'<tr><td>'+x.model+'</td><td>'+(x.shield?'ON':'off')+'</td><td>'+pct(x.success)+'</td><td style="color:'+(x.collision?'var(--bad)':'var(--ok)')+'">'+pct(x.collision)+'</td><td>'+pct(x.timeout)+'</td><td>'+(x.shield?pct(x.active):'–')+'</td></tr>').join('')+
  '</table><p class="muted">'+r.episodes+' unseen layouts per row. Full 300-layout result: collisions 0% for all 5 policies.</p>';busy(false)};
fetch('/api/info').then(r=>r.json()).then(i=>{if(!i.asr){$('mic').disabled=true;$('mic').title='speech off'}});
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-speech", action="store_true", help="skip Whisper and Piper (text commands only)")
    args = ap.parse_args()
    nlu, policy, runtime = load_stack(True)
    asr = tts = None
    if not args.no_speech:
        from edgebot.speech.asr import ASR
        from edgebot.speech.tts import TTS

        asr, tts = ASR("base"), TTS()
    STATE.update(nlu=nlu, policy=policy, runtime=runtime, asr=asr, tts=tts)
    STATE["bot"] = new_bot()
    print(f"EdgeBot control panel: http://localhost:{args.port}  (runtime: {runtime})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
