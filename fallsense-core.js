/* FallSense core algorithm - JavaScript port of fallsense/detector.py.
 * Same constants, same windows, same arithmetic (checked by scripts/parity_test.mjs).
 * Works in the browser (window.FallSenseCore) and in Node (module.exports).
 */
(function (root) {
  "use strict";
  const G = 9.80665;

  const CFG = {
    FS: 50.0, TRIG_G: 1.9, PEAK_S: 0.5, POST_S: 6.0, FF_G: 0.2, FF_DROP_S: 0.22,
    LOD_MOTION: 0.002, TILT_FALL_DEG: 40.0, VEL_FALL: 1.3, ACTIVE_FRAC: 0.35,
    T_REC: 30.0, T_PICK: 30.0, UPRIGHT_DEG: 35.0,
  };

  // ------------------------------------------------------------------ helpers
  function smv3(a) { return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }
  function pyRound(x) { // Python round-half-to-even (only matters on exact .5)
    const r = Math.round(x);
    if (Math.abs(x % 1) === 0.5) return 2 * Math.round(x / 2);
    return r;
  }
  function longestRun(mask) {
    let best = 0, cur = 0;
    for (const m of mask) { cur = m ? cur + 1 : 0; if (cur > best) best = cur; }
    return best;
  }
  function meanVec(rows) {
    const m = [0, 0, 0];
    if (!rows.length) return [NaN, NaN, NaN];
    for (const r of rows) { m[0] += r[0]; m[1] += r[1]; m[2] += r[2]; }
    return [m[0] / rows.length, m[1] / rows.length, m[2] / rows.length];
  }
  function motionRSS(rows) {
    if (rows.length < 3) return 0.0;
    const m = meanVec(rows);
    let v = 0;
    for (let k = 0; k < 3; k++) {
      let s = 0;
      for (const r of rows) { const d = r[k] - m[k]; s += d * d; }
      v += s / rows.length;
    }
    return Math.sqrt(v);
  }
  function angleDeg(u, v) {
    const c = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (Math.hypot(u[0], u[1], u[2]) * Math.hypot(v[0], v[1], v[2]) + 1e-12);
    return Math.acos(Math.max(-1, Math.min(1, c))) * 180 / Math.PI;
  }

  // ------------------------------------------------------------------ features
  // get(k) returns the 3-vector at absolute sample index k; n = number of samples available
  function eventFeatures(get, getW, n, i0, cfg) {
    const fs = cfg.FS;
    const win = (a, b) => {
      const lo = Math.max(0, i0 + pyRound(a * fs)), hi = Math.min(n, i0 + pyRound(b * fs));
      const out = [];
      for (let k = lo; k < hi; k++) out.push(get(k));
      return out;
    };
    const winS = (a, b) => win(a, b).map(smv3);
    const f = {};
    f.peak = Math.max(...winS(-0.1, 0.3));
    const pre = winS(-1.5, 0.0);
    f.min_pre = pre.length ? Math.min(...pre) : 1.0;
    f.ff_dur = longestRun(pre.map(v => v < cfg.FF_G)) / fs;
    const d = winS(-1.0, 0.0);
    let vs = 0; for (const v of d) vs += Math.max(1.0 - v, 0);
    f.v_pre = vs * G / fs;
    const pk = smv3(get(i0));
    const w = winS(-0.2, 0.2);
    f.imp_width = w.filter(v => v > 1.0 + 0.5 * (pk - 1.0)).length / fs;
    const b = winS(0.08, 1.0);
    f.bounce_ff = longestRun(b.map(v => v < cfg.FF_G)) >= Math.max(2, Math.trunc(0.04 * fs)) ? 1.0 : 0.0;
    const above = b.map(v => v > 1.5);
    let np_ = (above.length && above[0]) ? 1 : 0;
    for (let k = 1; k < above.length; k++) if (above[k] && !above[k - 1]) np_++;
    f.n_peaks = np_;
    f.still_early = motionRSS(win(1.2, 3.0));
    const post = win(2.0, cfg.POST_S);
    f.post_motion = motionRSS(post);
    const gPre = meanVec(win(-3.0, -1.5)), gPost = meanVec(post);
    f.dtheta = angleDeg(gPre, gPost);
    f.flat = Math.abs(gPost[2]) / (Math.hypot(gPost[0], gPost[1], gPost[2]) + 1e-12);
    const sp = post.map(smv3);
    f.active_frac = sp.length ? sp.filter(v => Math.abs(v - 1.0) > 0.12).length / sp.length : 0.0;
    f.pre_motion = motionRSS(win(-3.0, -1.5));
    if (getW) {
      const lo = Math.max(0, i0 + pyRound(-1.0 * fs)), hi = Math.min(n, i0 + pyRound(0.3 * fs));
      let gp = 0; for (let k = lo; k < hi; k++) gp = Math.max(gp, smv3(getW(k)));
      f.gyro_peak = gp;
      const lo2 = Math.max(0, i0 + pyRound(2.0 * fs)), hi2 = Math.min(n, i0 + pyRound(cfg.POST_S * fs));
      const ww = []; for (let k = lo2; k < hi2; k++) ww.push(getW(k));
      f.post_gyro = motionRSS(ww);
    } else { f.gyro_peak = NaN; f.post_gyro = NaN; }
    return f;
  }

  function classifyRules(f, cfg) {
    let drop = 0;
    drop += f.ff_dur >= cfg.FF_DROP_S ? 2 : 0;
    drop += f.bounce_ff > 0 ? 2 : 0;
    drop += f.still_early < cfg.LOD_MOTION ? 2 : 0;
    drop += f.imp_width <= 1.0 / cfg.FS + 1e-9 ? 1 : 0;
    drop += f.flat > 0.95 ? 1 : 0;
    let fall = 0;
    fall += f.peak >= 2.5 ? 1 : 0;
    fall += f.min_pre < 0.6 ? 1 : 0;
    fall += f.v_pre >= cfg.VEL_FALL ? 2 : 0;
    fall += f.dtheta >= cfg.TILT_FALL_DEG ? 2 : 0;
    fall += f.active_frac < cfg.ACTIVE_FRAC ? 1 : 0;
    const ev = { drop_score: drop, fall_score: fall };
    if (f.ff_dur >= cfg.FF_DROP_S && (f.bounce_ff > 0 || f.still_early < cfg.LOD_MOTION ||
        f.imp_width <= 1.0 / cfg.FS + 1e-9)) return ["DROP", ev];
    if (f.ff_dur >= cfg.FF_DROP_S && f.dtheta >= 30.0) return ["DROP", ev];
    if (drop >= 6) return ["DROP", ev];
    if (fall >= 6 && f.dtheta >= cfg.TILT_FALL_DEG && f.active_frac < cfg.ACTIVE_FRAC) return ["FALL", ev];
    return ["ADL", ev];
  }

  // ------------------------------------------------------------------ streaming detector
  class Detector {
    constructor(cfg) {
      this.cfg = Object.assign({}, CFG, cfg || {});
      this.A = []; this.W = []; this.base = 0; this.i = -1;
      this.state = "IDLE"; this.i0 = null; this.iUntil = 0; this.ev = null;
      this.events = [];
      this.onDecision = null; this.onOutcome = null;
    }
    get n() { return this.base + this.A.length; }
    a(k) { return this.A[k - this.base]; }
    w(k) { return this.W[k - this.base]; }
    smv(k) { return smv3(this.A[k - this.base]); }
    rows(lo, hi) { const o = []; for (let k = Math.max(lo, this.base); k < hi; k++) o.push(this.a(k)); return o; }

    push(acc, gyro) {           // acc in g, gyro in deg/s (or null) - one uniform 50 Hz sample
      this.A.push(acc); this.W.push(gyro || [0, 0, 0]);
      this.i++;
      if (this.A.length > 1500) { const cut = this.A.length - 1000; this.A.splice(0, cut); this.W.splice(0, cut); this.base += cut; }
      this.step(this.i);
    }

    step(i) {
      const cfg = this.cfg, fs = cfg.FS;
      const s = this.smv(i);
      if (this.state === "IDLE") {
        if (i >= this.iUntil && s >= cfg.TRIG_G && i > Math.trunc(3.0 * fs)) { this.state = "PEAK"; this.i0 = i; }
      } else if (this.state === "PEAK") {
        if (s > this.smv(this.i0)) this.i0 = i;
        if (i >= this.i0 + Math.trunc(cfg.PEAK_S * fs)) this.state = "WAIT";
      } else if (this.state === "WAIT") {
        if (i >= this.i0 + Math.trunc(cfg.POST_S * fs)) {
          const f = eventFeatures(k => this.a(k), this.useGyro ? (k => this.w(k)) : null, i + 1, this.i0, cfg);
          const [cls, evid] = classifyRules(f, cfg);
          const ev = { i_impact: this.i0, t_impact: this.i0 / fs, t_decision: i / fs, cls, feat: f, evidence: evid, outcome: null, t_outcome: null };
          this.ev = ev; this.events.push(ev);
          if (cls === "FALL") {
            this.state = "MON_FALL";
            ev.g_pre = meanVec(this.rows(Math.max(0, this.i0 - Math.trunc(3 * fs)), this.i0 - Math.trunc(1.5 * fs)));
          } else if (cls === "DROP") {
            this.state = "MON_DROP";
          } else {
            ev.outcome = "ADL"; this.state = "IDLE"; this.iUntil = this.i0 + Math.trunc(1.0 * fs);
          }
          if (this.onDecision) this.onDecision(ev);
          if (ev.outcome && this.onOutcome) this.onOutcome(ev);
        }
      } else if (this.state === "MON_FALL") {
        const ev = this.ev;
        const win = this.rows(i - Math.trunc(2 * fs) + 1, i + 1);
        const ang = angleDeg(meanVec(win), ev.g_pre);
        const mot = motionRSS(win);
        if ((ang < cfg.UPRIGHT_DEG && mot > 0.03) || motionRSS(this.rows(i - Math.trunc(3 * fs) + 1, i + 1)) > 0.15) {
          ev.outcome = "FALL_RECOVERED"; ev.t_outcome = i / fs; this.state = "IDLE"; this.iUntil = i + Math.trunc(2 * fs);
          if (this.onOutcome) this.onOutcome(ev);
        } else if (i >= ev.i_impact + Math.trunc((cfg.POST_S + cfg.T_REC) * fs)) {
          ev.outcome = "LONG_LIE_ALARM"; ev.t_outcome = i / fs; this.state = "ALARMED";
          if (this.onOutcome) this.onOutcome(ev);
        }
      } else if (this.state === "MON_DROP") {
        const ev = this.ev;
        if (!ev.g_rest) ev.g_rest = meanVec(this.rows(ev.i_impact + Math.trunc(1.2 * fs), ev.i_impact + Math.trunc(3.0 * fs)));
        const win = this.rows(i - Math.trunc(1 * fs) + 1, i + 1);
        const moved = angleDeg(meanVec(win), ev.g_rest), mot = motionRSS(win);
        // picked up = large motion, or held in a hand: above the LOD and no longer in the resting orientation
        if (mot > 0.05 || (mot > 4 * cfg.LOD_MOTION && moved > 15.0)) {
          ev.outcome = "DROP_PICKED_UP"; ev.t_outcome = i / fs; this.state = "IDLE"; this.iUntil = i + Math.trunc(2 * fs);
          if (this.onOutcome) this.onOutcome(ev);
        } else if (i >= ev.i_impact + Math.trunc((cfg.POST_S + cfg.T_PICK) * fs)) {
          ev.outcome = "UNATTENDED_DROP_CHECKIN"; ev.t_outcome = i / fs; this.state = "ALARMED";
          if (this.onOutcome) this.onOutcome(ev);
        }
      }
    }
    resetAfterAlarm() { this.state = "IDLE"; this.iUntil = this.i + Math.trunc(2 * this.cfg.FS); }
    finish() {
      const ev = this.ev;
      if (ev && ev.outcome === null) ev.outcome = { FALL: "FALL_PENDING", DROP: "DROP_PENDING" }[ev.cls] || "ADL";
      return this.events;
    }
  }

  // ------------------------------------------------------------------ uniform resampler
  // Browser devicemotion events arrive at ~60 Hz with jitter: linear interpolation onto 50 Hz.
  class Resampler {
    constructor(fs, onSample) { this.dt = 1000 / fs; this.on = onSample; this.prev = null; this.tNext = null; this.gaps = 0; }
    push(tMs, v) {
      if (this.prev && (tMs - this.prev.t > 250 || tMs <= this.prev.t)) { // tab paused / clock jump
        if (tMs - this.prev.t > 250) this.gaps++;
        this.prev = null;
      }
      if (!this.prev) { this.prev = { t: tMs, v }; this.tNext = tMs; return; }
      while (this.tNext <= tMs) {
        const u = (this.tNext - this.prev.t) / (tMs - this.prev.t);
        this.on(this.prev.v.map((p, k) => p + u * (v[k] - p)), this.tNext);
        this.tNext += this.dt;
      }
      this.prev = { t: tMs, v };
    }
  }

  // ------------------------------------------------------------------ blank test (LOD)
  function lodFromBlank(rows, fs, winS, k) {
    winS = winS || 1.8; k = k || 3.3;
    const n = Math.trunc(winS * fs), step = Math.trunc(n / 2), vals = [];
    for (let i = 0; i + n <= rows.length - 0; i += step) { if (i >= rows.length - n) break; vals.push(motionRSS(rows.slice(i, i + n))); }
    const m = vals.reduce((a, b) => a + b, 0) / vals.length;
    const sd = Math.sqrt(vals.reduce((a, b) => a + (b - m) * (b - m), 0) / vals.length);
    return { mean_blank: m, sd_blank: sd, lod: m + k * sd, n_windows: vals.length };
  }

  const api = { CFG, G, Detector, Resampler, eventFeatures, classifyRules, motionRSS, lodFromBlank, smv3 };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.FallSenseCore = api;
})(typeof window !== "undefined" ? window : globalThis);
