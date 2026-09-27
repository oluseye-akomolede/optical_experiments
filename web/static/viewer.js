// MSE mode viewer: the object's field as a rotatable 3-D volume.
//
// Data comes from four endpoints, all same-origin:
//   /jobs/<id>/status            small poll for progress
//   /jobs/<id>/index.json        the mode index: one entry per mode, no field
//   /jobs/<id>/field/<n>.bin     one mode's vector field, float16 LE, gzipped
//   /jobs/<id>/files/...         proxied results (modes.json, surface.glb, .vtu)
//
// The split between the index and the per-mode payload is the point: the index
// is fetched once and is small, and a mode's field data (which is large) is
// fetched only when that mode is first shown. See modecache.js.
//
// The field is a raymarched volume texture rather than slices: the owner asked
// for a smoke plot they can rotate, showing a *component* they choose rather
// than a stack of planes. The vector-to-scalar step lives in components.js, and
// the field bytes drop straight into a HalfFloat Data3DTexture with no
// reshaping -- the solver's (nz, ny, nx, 3) order is already the texture's.
//
// three.js is vendored under /static/vendor (pinned 0.160.0, MIT); nothing is
// fetched from a CDN, because the app is only reachable on the tailnet.

import * as THREE from './vendor/three.module.min.js';
import { OrbitControls } from './vendor/OrbitControls.js';
import { GLTFLoader } from './vendor/GLTFLoader.js';
import { wantsFetch, modeFieldUrl, storeFields } from './modecache.js';
import { boxCentre, componentShare, componentVolume, float32ToHalf,
         halfToFloat32, SYSTEM_NAMES,
         SYSTEM_LABELS, COMPONENT_NAMES, COMPONENT_LABELS, DEFAULT_SYSTEM,
         DEFAULT_COMPONENT }
  from './components.js';

const root = document.getElementById('job');
if (root) init(root);

// set by init() once the field data is in, so the table rows can select a mode
let selectModeRef = null;

function init(root) {
  const jobId = root.dataset.jobId;
  const status = document.getElementById('status-text');
  const stage = document.getElementById('stage-text');
  const bar = document.getElementById('bar');
  const elapsed = document.getElementById('elapsed');
  const queueLine = document.getElementById('queue-line');
  const progress = document.getElementById('progress');
  const failure = document.getElementById('failure');
  const failureText = document.getElementById('failure-text');
  const viewer = document.getElementById('viewer');
  const modesCard = document.getElementById('modes-card');
  const modesBody = document.getElementById('modes-body');
  const started = Date.now();

  const STAGE_FRACTION = {
    queued: 0.03, meshing: 0.25, assembling: 0.5, solving: 0.75, writing: 0.92,
    done: 1,
  };

  function tick() {
    fetch(`/jobs/${jobId}/status`, { cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.status))))
      .then(render)
      .catch(() => { /* transient; the next tick retries */ });
  }

  function render(data) {
    status.textContent = data.status;
    stage.textContent = data.stage && data.stage !== data.status ? data.stage : '';
    bar.style.width = `${(STAGE_FRACTION[data.stage || data.status] || 0) * 100}%`;
    elapsed.textContent = `elapsed ${formatSeconds((Date.now() - started) / 1000)}`;

    if (data.queue_position) {
      queueLine.hidden = false;
      queueLine.textContent = data.queue_position === 1
        ? 'Next in line — starting shortly.'
        : `Waiting in the queue: position ${data.queue_position}.`;
    } else {
      queueLine.hidden = true;
    }

    if (data.status === 'failed') {
      progress.hidden = true;
      failure.hidden = false;
      failureText.textContent = data.message
        || 'Something went wrong while computing the modes.';
      return;
    }

    if (data.status === 'done') {
      progress.hidden = true;
      failure.hidden = true;
      finish(data);
      return;
    }
    setTimeout(tick, 3000);
  }

  let loaded = false;
  function finish(data) {
    buildModeTable(data.modes || []);
    if (loaded) return;
    loaded = true;
    viewer.hidden = false;
    modesCard.hidden = false;
    loadFieldData(jobId, data);
  }

  tick();
}

// --------------------------------------------------------------------------
// Volume rendering
// --------------------------------------------------------------------------

// Two ramps, because a signed component and a magnitude are two different
// quantities and one ramp cannot say both. Each is declared once here and
// mirrored in the shader's colour function and the legend bar, so the key
// beside the plot is guaranteed to be the ramp the plot is using.
//
// The design calls for it: "signed components use a diverging map (red +,
// blue -, transparent near 0); |E| uses a sequential map". The first cut used
// the sequential ramp for everything and said "signed" by multiplying the
// colour by sign(v) -- which does not produce a cool end, it produces a
// *negative* colour, and negative RGB clamps to black at the framebuffer. So
// every negative lobe rendered as a black half-disc: not a wrong hue, an
// absent one. That is the defect the render check was supposed to catch and
// did not, because the check only asked whether *some* pixels were coloured.
//
// The diverging ramp goes blue -> near-transparent white -> red. The middle
// stop is deliberately pale rather than saturated: it is where the colour
// carries no sign, so it must not read as either lobe, and the alpha fade near
// zero (see uThreshold) is what keeps "no field here" from being drawn as a
// third colour.
const COLORMAP_SEQUENTIAL = [
  [0.00, 0.05, 0.17, 0.42],
  [0.35, 0.12, 0.54, 0.84],
  [0.60, 0.26, 0.82, 0.49],
  [0.82, 1.00, 0.78, 0.34],
  [1.00, 1.00, 0.35, 0.24],
];

// -1 at the cool end, 0 at the pale middle, +1 at the warm end.
const COLORMAP_DIVERGING = [
  [-1.00, 0.15, 0.35, 0.85],
  [-0.45, 0.25, 0.60, 0.95],
  [0.00, 0.92, 0.92, 0.90],
  [0.45, 0.98, 0.60, 0.22],
  [1.00, 0.80, 0.16, 0.14],
];

// The box's vertex coordinates run 0..1 on every face and hit all eight
// corners, so
// it is the unit-cube coordinate the raymarch needs. It is also used here to
// place the box in the scene: a half-size box centred on the origin, scaled by
// the mesh to the grid's physical extent. Scaling a unit geometry is what keeps
// local space equal to texture space, whatever the grid's aspect ratio.
// The box is the unit cube, and `vLocal` is derived from the geometry's own
// vertex position rather than from `uv`. The two are *not* interchangeable and
// the difference is the whole bug this comment exists for: `uv` on a
// BoxGeometry runs 0..1 per face, which is what the raymarch wants, but the
// position runs -0.5..0.5, so a fragment's `vLocal` would name a point that is
// not the one the geometry actually occupies. Every ray would then test a slab
// it is not inside and discard, and the plot would be a blank canvas with no
// error anywhere. Taking the coordinate from `position` makes geometry and
// shader agree by construction, whatever size the mesh is scaled to.
const VERTEX_SHADER = `
varying vec3 vLocal;
void main() {
  vLocal = position + 0.5;   // the unit cube's -0.5..0.5 becomes 0..1
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

// `cameraPosition` is three.js's world-space camera, but everything else in the
// fragment shader is in the box's own 0..1 space. Translating the camera by the
// box's centre and dividing by its size is the inverse of the mesh's own
// transform, so the ray is marched in the same frame as `vLocal`. Skipping this
// works only for a box at the origin with unit scale -- with a real grid the
// camera sits offset and the ray misses or grazes the volume.
const CAMERA_LOCAL = `
uniform vec3 uBoxCentre;
uniform vec3 uBoxSize;
vec3 cameraLocal() {
  return (cameraPosition - uBoxCentre) / max(uBoxSize, vec3(1e-12));
}
`;

// No `#version` line here, and that is deliberate rather than an omission.
// `glslVersion: THREE.GLSL3` makes three.js prepend its own `#version 300 es`
// ahead of a ~55-line block of shared declarations, because a ShaderMaterial
// is not a raw one. A `#version` written here would therefore land *after*
// that block, which GLSL forbids -- "'#version' directive must occur before
// anything else" -- and the fragment shader would fail to compile. three.js
// logs that to the console and then draws nothing, so the plot is a blank
// canvas with no exception and no failing unit test; only the render check
// could see it. The `in`/`out`/`texelFetch` syntax below still needs GLSL3,
// which is exactly what `glslVersion` is asking for.
const FRAGMENT_SHADER = `
precision highp float;
precision highp sampler3D;

${CAMERA_LOCAL}

uniform sampler3D uField;
uniform vec3 uVolumeSize;   // texels per axis, for the sampling step
uniform float uAlphaScale;
uniform float uThreshold;
uniform bool uSigned;       // false for |E|, which has no negative half
in vec3 vLocal;
out vec4 fragColor;

// Sequential ramp, for |E|. Mirrors COLORMAP_SEQUENTIAL in viewer.js, whose
// table the legend gradient is drawn from, so the key beside the plot is the
// ramp the plot is actually using.
const vec3 CM0 = vec3(0.05, 0.17, 0.42);
const vec3 CM1 = vec3(0.12, 0.54, 0.84);
const vec3 CM2 = vec3(0.26, 0.82, 0.49);
const vec3 CM3 = vec3(1.00, 0.78, 0.34);
const vec3 CM4 = vec3(1.00, 0.35, 0.24);

vec3 colormapSequential(float t) {
  t = clamp(t, 0.0, 1.0);
  if (t < 0.35) return mix(CM0, CM1, t / 0.35);
  if (t < 0.60) return mix(CM1, CM2, (t - 0.35) / 0.25);
  if (t < 0.82) return mix(CM2, CM3, (t - 0.60) / 0.22);
  return mix(CM3, CM4, (t - 0.82) / 0.18);
}

// Diverging ramp, for a signed component. Blue at -1, pale at 0, red at +1.
// Mirrors COLORMAP_DIVERGING in viewer.js.
const vec3 DV0 = vec3(0.15, 0.35, 0.85);
const vec3 DV1 = vec3(0.25, 0.60, 0.95);
const vec3 DV2 = vec3(0.92, 0.92, 0.90);
const vec3 DV3 = vec3(0.98, 0.60, 0.22);
const vec3 DV4 = vec3(0.80, 0.16, 0.14);

// The argument is the SIGNED value in -1..1, NOT its magnitude: the sign picks
// the half of the ramp. This is the function whose absence produced the black
// half-disc -- the previous code took colormap(|v|) and multiplied the colour
// by sign(v), and multiplying a colour by a negative number clamps to black.
//
// (No backticks anywhere in this shader source: it is a JS template literal, so
// one would end the string early and the file would fail to parse -- as it did
// when this comment was first written.)
vec3 colormapDiverging(float s) {
  float t = clamp(s, -1.0, 1.0) * 0.5 + 0.5;   // -1..1 -> 0..1
  if (t < 0.25) return mix(DV0, DV1, t / 0.25);
  if (t < 0.50) return mix(DV1, DV2, (t - 0.25) / 0.25);
  if (t < 0.75) return mix(DV2, DV3, (t - 0.50) / 0.25);
  return mix(DV3, DV4, (t - 0.75) / 0.25);
}

bool isNan(float x) { return !(x == x); }

// One texel, with "no sample here" read as zero.
//
// A NaN texel means the solver wrote "outside the object" there, so it must
// contribute no colour. Letting the hardware interpolate it instead does not
// work, and this is the whole reason for this function: NaN in the filtered
// value is not the same as NaN at a corner, and a trilinear sample that is
// merely near a NaN corner comes back NaN in some drivers and in others comes
// back a plausible-looking value that is not a field. Either way the picture is
// drawn by the driver's NaN policy rather than by the data. So the corners are
// tested where the corner is unambiguous -- at a texel centre, which is exactly
// what texelFetch returns -- and any NaN corner sends the whole sample to the
// nearest-neighbour value, a real sample rather than an average.
float sampleTexel(ivec3 c) {
  vec4 s = texelFetch(uField, c, 0);
  return (s.a > 0.0 && !isNan(s.r)) ? s.r : 0.0;
}

float sampleField(vec3 uvw) {
  ivec3 n = ivec3(uVolumeSize);
  vec3 texel = uvw * vec3(n) - 0.5;
  ivec3 lo = ivec3(floor(texel));
  vec3 f = texel - vec3(lo);

  vec3 t000 = vec3(lo + ivec3(0, 0, 0));
  vec3 t100 = vec3(lo + ivec3(1, 0, 0));
  vec3 t010 = vec3(lo + ivec3(0, 1, 0));
  vec3 t110 = vec3(lo + ivec3(1, 1, 0));
  vec3 t001 = vec3(lo + ivec3(0, 0, 1));
  vec3 t101 = vec3(lo + ivec3(1, 0, 1));
  vec3 t011 = vec3(lo + ivec3(0, 1, 1));
  vec3 t111 = vec3(lo + ivec3(1, 1, 1));

  bool bad = any(lessThan(t000, vec3(0.0))) || any(lessThan(t100, vec3(0.0))) ||
             any(lessThan(t010, vec3(0.0))) || any(lessThan(t110, vec3(0.0))) ||
             any(lessThan(t001, vec3(0.0))) || any(lessThan(t101, vec3(0.0))) ||
             any(lessThan(t011, vec3(0.0))) || any(lessThan(t111, vec3(0.0))) ||
             any(greaterThanEqual(t000, vec3(n))) ||
             any(greaterThanEqual(t100, vec3(n))) ||
             any(greaterThanEqual(t010, vec3(n))) ||
             any(greaterThanEqual(t110, vec3(n))) ||
             any(greaterThanEqual(t001, vec3(n))) ||
             any(greaterThanEqual(t101, vec3(n))) ||
             any(greaterThanEqual(t011, vec3(n))) ||
             any(greaterThanEqual(t111, vec3(n)));

  float v000 = sampleTexel(ivec3(t000));
  float v100 = sampleTexel(ivec3(t100));
  float v010 = sampleTexel(ivec3(t010));
  float v110 = sampleTexel(ivec3(t110));
  float v001 = sampleTexel(ivec3(t001));
  float v101 = sampleTexel(ivec3(t101));
  float v011 = sampleTexel(ivec3(t011));
  float v111 = sampleTexel(ivec3(t111));

  if (bad || isNan(v000) || isNan(v100) || isNan(v010) || isNan(v110) ||
      isNan(v001) || isNan(v101) || isNan(v011) || isNan(v111)) {
    // Nearest texel, clamped into the texture.
    //
    // The clamp is defensive, and the reason it is not load-bearing is worth
    // writing down, because the obvious reading of this line is wrong. At the
    // box's own high face uvw == 1.0, floor(uvw * n) is exactly n -- one past
    // the last texel -- and an out-of-range texelFetch is undefined in WebGL2.
    // But the raymarch cannot put uvw there: it reads sample i at offset
    // (i + jitter) * dt with i <= steps-1 and jitter < 1, so the largest offset
    // read is strictly less than steps * dt = span, and the largest uvw the
    // march reads on the live grid is 0.999999994 -- index 31 of 0..31, in
    // range. The overshoot to 1.0056 does happen, but at the p += rd * dt
    // advance after the final read, whose position is never sampled; the guard
    // is at the top of the loop and the advance is at the bottom, and that
    // ordering is what bounds it.
    //
    // The window is zero-width, not absent, and the difference is worth stating
    // because it is easy to test this wrongly: floor(uvw * n) >= n first happens
    // at uvw == 1.0 EXACTLY, so a harness that sweeps uvw and includes 1.0 will
    // report out-of-range hits and mean nothing by it -- that mistake was made
    // here once, and reported as 481201 samples before being retracted. The
    // correct statement is not "it never fires" but "there is no uvw < 1.0 at
    // which it fires, and the march does not reach 1.0".
    //
    // So nothing observable turns on this clamp, and it is not a fix for the
    // cloud's flat face -- that is the object's own silhouette (a 10x7x5mm box,
    // constant x- and y-extent across every z-slice of the finite mask). It is
    // kept because it makes the fallback provably in-range on every driver for
    // two tokens, and because a bound that holds by reasoning about a caller
    // elsewhere in the file is worth pinning where the fetch happens.
    ivec3 c = ivec3(floor(uvw * vec3(n)));
    return sampleTexel(clamp(c, ivec3(0), n - ivec3(1)));
  }

  float x00 = mix(v000, v100, f.x);
  float x10 = mix(v010, v110, f.x);
  float x01 = mix(v001, v101, f.x);
  float x11 = mix(v011, v111, f.x);
  float y0 = mix(x00, x10, f.y);
  float y1 = mix(x01, x11, f.y);
  return mix(y0, y1, f.z);
}

void main() {
  // The whole raymarch runs in local space: the box spans 0..1 there, so the
  // slab test and the texture lookup need no physical units at all. Only the
  // step length depends on the grid's texel counts.
  //
  // The direction is the camera *to* the fragment, and the march runs backwards
  // along it. This is what side: BackSide requires -- the fragment being shaded
  // is the box's far wall, so the ray has to travel from that far wall back
  // toward the eye to cross the volume at all. Marching the other way, with the
  // direction pointing away from the camera, leaves the box on the first step,
  // so the slab test rejects every fragment and the canvas comes out empty.
  // That failure is invisible without a render check: the shader compiles,
  // nothing throws, and a blank plot looks like a mode with no field.
  vec3 ro = vLocal;
  vec3 rd = normalize(cameraLocal() - vLocal);
  vec3 inv = 1.0 / rd;
  vec3 t0 = (vec3(0.0) - ro) * inv;
  vec3 t1 = (vec3(1.0) - ro) * inv;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);
  float tnear = max(max(tmin.x, tmin.y), tmin.z);
  float tfar = min(min(tmax.x, tmax.y), tmax.z);
  if (tfar <= 0.0 || tfar <= tnear) discard;   // box behind, or missed

  vec3 size = max(uVolumeSize, vec3(1.0));
  float steps = clamp(2.0 * length(size), 64.0, 768.0);
  float span = tfar - max(tnear, 0.0);
  float dt = span / steps;

  // A per-pixel offset into the first step, so the banding of a fixed-step
  // raymarch is traded for noise. The volume's own noise is not the point --
  // the owner is reading a mode's shape -- so this is deliberately only half a
  // step: enough to break the bands, not enough to look like a defect.
  float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);

  vec4 acc = vec4(0.0);
  vec3 p = ro + rd * (max(tnear, 0.0) + dt * jitter);
  for (int i = 0; i < 768; i++) {
    if (float(i) >= steps || acc.a > 0.995) break;
    float v = sampleField(p);
    // The value is already zero wherever the solver wrote NaN, so nothing else
    // has to test for it here: no colour is built from a NaN, and the sample
    // still advances the ray, so the hole in the object stays a hole.
    float t = uSigned ? abs(v) : v;
    if (t > uThreshold) {
      // A signed component takes the diverging ramp and is handed its *signed*
      // value, so the negative lobe gets the cool end and the positive lobe the
      // warm end. Without this, Ez and |Ez| would be the same picture, and a
      // standing wave's two out-of-phase lobes -- the thing the component view
      // exists to show -- would be indistinguishable from each other.
      //
      // The value is normalised by the component's own peak before it reaches
      // here -- see the texture build, which divides by it -- so "1.0" in the
      // diverging ramp is the strongest sample anywhere in the picture, the two
      // lobes are directly comparable, and uThreshold means the same contrast
      // floor in every mode. (It did not, until the build learned to divide: an
      // absolute threshold on an unnormalised component made a faint mode draw
      // its faint structure as though there were none.) This is why the ramp
      // takes a signed argument
      // rather than a magnitude plus a sign separately: the two lobes must
      // share one scale, or a slightly-positive and a strongly-negative region
      // would draw the same colour.
      vec3 c = uSigned ? colormapDiverging(v) : colormapSequential(t);
      // Alpha is accumulated as an integral along the ray, which is what makes
      // the picture a smoke plot of the mode's cross-section rather than a
      // function of whichever face the camera happens to be looking at.
      //
      // The step is measured as a FRACTION OF THE BOX, not in metres. An
      // earlier version multiplied by uExtent -- the box's physical size -- on
      // the theory that optical depth should be physical. It is not, and the
      // effect was that the same field drew at wildly different opacities
      // depending on how big the object happened to be: on the live job
      // (10.4 mm box, 169 steps) the per-step alpha came to 2.5e-4, so the whole
      // mode accumulated 4% opacity and the plot was invisible -- every
      // component, including |E|. The synthetic fixture this was first tried on
      // had a box large relative to its grid, so it looked fine there and the
      // bug stayed hidden until real data arrived.
      //
      // dt is already in the box's own 0..1 units, so t * uAlphaScale * dt
      // says "how much of this ray's length lies in field of this strength".
      // A 5 mm object and a 5 m one then draw identically, which is what the
      // owner wants to see: the mode's shape, not its size.
      float a = clamp(t * uAlphaScale * dt, 0.0, 1.0);
      acc.rgb += (1.0 - acc.a) * a * c;
      acc.a += (1.0 - acc.a) * a;
    }
    p += rd * dt;
  }
  if (acc.a <= 0.002) discard;
  fragColor = acc;
}
`;

function loadFieldData(jobId, statusData) {
  const canvas = document.getElementById('scene');
  const fallback = document.getElementById('viewer-fallback');

  // WebGL2 is not optional here: a HalfFloat 3D texture needs it, and so does
  // the shader. A browser without it gets an explanation rather than a blank
  // black rectangle that looks like a job with no field.
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false,
                                             alpha: false, powerPreference: 'high-performance' });
  const gl = renderer.getContext();
  if (!gl || !(gl instanceof WebGL2RenderingContext)) {
    canvas.hidden = true;
    if (fallback) fallback.hidden = false;
    return;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0e12);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);

  // The plot is unlit -- the colour comes from the shader, and the volume is a
  // transparent object -- so no lights are added and the surface (if any) uses
  // a flat material.

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.zoomSpeed = 0.8;

  let surfaceMesh = null;

  // The box the raymarch runs in: the unit cube centred on the origin, scaled
  // to the grid's physical extent and moved to its centre. Local space is
  // therefore 0..1 across the grid however the box is scaled, which is what
  // lets the shader work in grid coordinates and never in metres. The vertex
  // shader recovers that 0..1 from the geometry's own vertices -- see
  // VERTEX_SHADER, where deriving it from `uv` instead made every ray miss.
  const boxGeometry = new THREE.BoxGeometry(1, 1, 1);
  // Back faces are rendered on purpose: the camera may sit inside the box, and
  // a front-face-only box would vanish the moment the user zoomed in past the
  // near wall, which reads as the plot having disappeared.
  const volumeMaterial = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: VERTEX_SHADER,
    fragmentShader: FRAGMENT_SHADER,
    uniforms: {
      uField: { value: null },
      uVolumeSize: { value: new THREE.Vector3(1, 1, 1) },
      uAlphaScale: { value: 4.0 },
      uThreshold: { value: 0.02 },
      // Multiplied by the box's largest side once the grid is known, so the
      // emissive integral is exp(-k * t * depth) with depth in physical units.
      uSigned: { value: true },
      uBoxCentre: { value: new THREE.Vector3() },
      uBoxSize: { value: new THREE.Vector3(1, 1, 1) },
    },
    side: THREE.BackSide,
    transparent: true,
    depthWrite: false,
  });
  const volume = new THREE.Mesh(boxGeometry, volumeMaterial);
  volume.visible = false;
  scene.add(volume);

  let manifest = null;
  let modes = [];
  let current = 0;
  let bounds = null;
  let system = DEFAULT_SYSTEM;
  let name = DEFAULT_COMPONENT;
  let texture = null;
  // The share of |E| the chosen component reaches in the mode on screen, in
  // 0..1, or null when it could not be measured. Held here so the mode caption
  // toggling and the component caption cannot disagree: both read it, and it is
  // set at exactly the one place the picture is rebuilt.
  let volumeShare = null;

  function boundsOf(grid) {
    const lo = grid.origin.slice();
    const n = [grid.nx, grid.ny, grid.nz];
    const hi = [0, 0, 0];
    // grid.nx/ny/nz are counts along the x/y/z axes; origin/spacing are xyz
    // ordered too, so the corner is a plain elementwise product
    for (let i = 0; i < 3; i += 1) {
      hi[i] = lo[i] + grid.spacing[i] * (n[i] - 1);
    }
    return { lo, hi, centre: lo.map((v, i) => (v + hi[i]) / 2),
             size: lo.map((v, i) => hi[i] - v) };
  }

  function fitVolume() {
    volume.scale.set(Math.max(bounds.size[0], 1e-9),
                     Math.max(bounds.size[1], 1e-9),
                     Math.max(bounds.size[2], 1e-9));
    volume.position.set(bounds.centre[0], bounds.centre[1], bounds.centre[2]);
    const grid = manifest.grid;
    volumeMaterial.uniforms.uVolumeSize.value.set(
      Math.max(grid.nx, 1), Math.max(grid.ny, 1), Math.max(grid.nz, 1));
    // uBoxCentre/uBoxSize are the only place the box's physical size is still
    // needed, and only to put the camera into the shader's local space -- the
    // raymarch's own step is a fraction of the box (see the alpha line), so the
    // plot looks the same whatever the object's real size.
    volumeMaterial.uniforms.uBoxCentre.value.set(bounds.centre[0],
                                                 bounds.centre[1],
                                                 bounds.centre[2]);
    volumeMaterial.uniforms.uBoxSize.value.set(
      Math.max(bounds.size[0], 1e-12), Math.max(bounds.size[1], 1e-12),
      Math.max(bounds.size[2], 1e-12));
  }

  function frameCamera() {
    // Distance from the box centre to the eye, as a multiple of the box's
    // largest side. The first cut used a "radius" as if it were a distance-to-
    // corner and then added an offset in x, y AND z -- so the eye sat at
    // sqrt(1 + 0.64 + 1) * 1.6 = 2.87 box-widths out while the object spans one,
    // and the mode rendered as a small dot in the middle of the frame. The
    // object fills ~84% of the grid box (measured on the live job), so any
    // distance past roughly 2 box-widths leaves most of the canvas empty.
    //
    // 1.35 puts the near face of the box just inside the frustum and the object
    // comfortably filling the frame, with a little air around it so the outline
    // and a rotating object do not touch the edge. The diagonal view keeps all
    // three axes legible from the first frame.
    const distance = (Math.max(...bounds.size) || 1) * 1.35;
    const unit = 1 / Math.sqrt(3);   // the (1,1,1) direction, normalised
    camera.position.set(bounds.centre[0] + distance * unit,
                        bounds.centre[1] + distance * unit,
                        bounds.centre[2] + distance * unit);
    camera.near = Math.max(distance / 100, 1e-6);
    camera.far = distance * 100;
    camera.updateProjectionMatrix();
    controls.target.set(bounds.centre[0], bounds.centre[1], bounds.centre[2]);
    controls.update();
  }

  function addSurface(id) {
    fetch(`/jobs/${id}/files/surface.glb`)
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(r.status))))
      .then((buffer) => new Promise((resolve, reject) => {
        new GLTFLoader().parse(buffer, '', resolve, reject);
      }))
      .then((gltf) => {
        const material = new THREE.MeshBasicMaterial({
          color: 0x9fb6cc, transparent: true, opacity: 0.08,
          side: THREE.DoubleSide, depthWrite: false,
        });
        gltf.scene.traverse((child) => {
          if (child.isMesh) {
            child.material = material;
            surfaceMesh = child;
            scene.add(child);
          }
        });
      })
      .catch(() => { /* the field is the point; a missing surface is survivable */ });
  }

  // -- selectors ----------------------------------------------------------
  function buildSelectors() {
    const systemSelect = document.getElementById('component-system');
    const nameSelect = document.getElementById('component-name');
    if (!systemSelect || !nameSelect) return;
    for (const s of SYSTEM_NAMES) {
      const option = document.createElement('option');
      option.value = s;
      option.textContent = SYSTEM_LABELS[s] || s;
      systemSelect.appendChild(option);
    }
    systemSelect.value = system;
    fillComponents(nameSelect);
    systemSelect.addEventListener('change', () => {
      system = systemSelect.value;
      fillComponents(nameSelect);
      refreshVolume();
    });
    nameSelect.addEventListener('change', () => {
      name = nameSelect.value;
      refreshVolume();
    });
  }

  function fillComponents(nameSelect) {
    const names = COMPONENT_NAMES(system);
    nameSelect.innerHTML = '';
    for (const n of names) {
      const option = document.createElement('option');
      option.value = n;
      option.textContent = COMPONENT_LABELS[n] || n;
      nameSelect.appendChild(option);
    }
    // A component name belongs to exactly one system, so switching system
    // always moves the selection; the fallback keeps the picker valid if a
    // system ever omits the name the previous one chose.
    if (!names.includes(name)) name = names[0];
    nameSelect.value = name;
  }

  // -- legend -------------------------------------------------------------
  // Drawn from the same table the shader hard-codes, so the bar beside the plot
  // is the ramp in the plot rather than a hand-matched guess -- and from the
  // *matching* table, because the two ramps say different things. A signed
  // component runs blue at -max through the pale zero crossing to red at +max;
  // |E| runs the sequential ramp from 0 to its maximum.
  //
  // The previous version drew one ramp and dimmed its left half by a factor of
  // 0.45 to stand in for "negative". That is the legend being wrong in the same
  // way the plot was: it made the negative lobe read as *the same hue, darker*
  // when it should be a different hue, so even a reader who looked at the key
  // would have taken a black half-disc for a weak one.
  function paintLegend() {
    const bar = document.getElementById('legend-bar');
    if (!bar) return;
    const signed = name !== 'E';
    const table = signed ? COLORMAP_DIVERGING : COLORMAP_SEQUENTIAL;
    const stops = [];
    const n = 64;
    for (let i = 0; i <= n; i += 1) {
      const u = i / n;
      // Diverging: u = 0 at -max, 0.5 at zero, 1 at +max, so the ramp's own
      // coordinate is 2u - 1. Sequential: u is already the ramp's coordinate.
      const [r, g, b] = rampAt(signed ? 2 * u - 1 : u, table);
      stops.push(`rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ` +
                 `${Math.round(b * 255)}) ${(u * 100).toFixed(1)}%`);
    }
    bar.style.background = `linear-gradient(90deg, ${stops.join(', ')})`;
    const label = COMPONENT_LABELS[name] || name;
    const note = document.getElementById('legend-note');
    if (note) {
      note.textContent = signed
        ? `Colour is ${label}, signed: the cool end is negative, the warm end ` +
          'positive, and the pale middle is the zero crossing. On the left is ' +
          'the end colour dimmed, so a negative lobe reads as dim and cool ' +
          'rather than as a different quantity. Every mode is normalised to ' +
          'max |E| = 1, so the scale is the same for all of them. Transparent ' +
          'is outside the object.'
        : 'Colour is |E|, normalised to 1 and the same scale for every mode. ' +
          'Transparent is outside the object.';
    }
    const lo = document.getElementById('legend-lo');
    const hi = document.getElementById('legend-hi');
    if (lo) lo.textContent = signed ? `−${label}` : '0';
    if (hi) hi.textContent = signed ? `+${label}` : `${label} max`;
    paintComponentNote(signed);
  }

  // The caption that tells a faint component apart from a broken one.
  //
  // It exists because of a specific misreading -- mine, corrected by the lead.
  // Mode 1's cylindrical Erho draws as an almost empty frame, and an almost
  // empty frame is what a render bug looks like, so I first read it as one and
  // proposed rescaling the component. It is not a bug: mode 1 is the
  // z-polarised (1,1,0) mode, and its Erho reaches only 8% of the mode's own
  // peak |E|, so a faint cloud is the honest picture. Measured in this build:
  // 31 field pixels of 294402. Mode 2's Erho reaches 98% and draws strongly.
  // Same control, opposite pictures, both correct -- so the fix is not to
  // change the scale (which would inflate a small component into a confident
  // lie) but to say plainly which case the viewer is looking at.
  //
  // The number comes from componentShare() off the same decoded field the
  // picture is built from, so the caption can never describe a different field
  // from the one on screen. When the share cannot be measured the caption says
  // so rather than printing a plausible number -- a caption that reads like a
  // measurement when it is not one is the could-not-ask defect wearing a
  // caption's clothes.
  const SMALL_SHARE = 0.2;

  function percent(share) {
    return `${Math.round(share * 100)}%`;
  }

  function paintComponentNote(signed) {
    const note = document.getElementById('component-note');
    if (!note) return;
    const label = COMPONENT_LABELS[name] || name;

    // |E| is the mode's own peak by definition; there is nothing to qualify.
    if (!signed || name === 'E') { note.hidden = true; return; }
    note.hidden = false;

    if (volumeShare === null || !Number.isFinite(volumeShare)) {
      note.textContent =
        `Could not measure how much ${label} contributes to this mode, so ` +
        'this picture is not evidence that the component is small.';
      return;
    }

    const share = percent(volumeShare);
    if (volumeShare < SMALL_SHARE) {
      // The whole point: name the smallness, in the picture's own terms, so a
      // viewer does not read the faintness as a fault. The reason is physical
      // -- a mode polarised the other way is small in this component.
      note.textContent =
        `${label} is small in this mode: it reaches only ${share} of the ` +
        `mode's peak |E|, so this component draws as a faint cloud. The same ` +
        `control is strong in other modes. The bar below is unchanged -- it ` +
        `is the shared ${label} scale for every mode, not a scale for this ` +
        'component alone.';
    } else {
      note.textContent =
        `${label} reaches ${share} of this mode's peak |E|, so it draws on ` +
        'the shared scale and should read as strongly as |E| does.';
    }
  }

  // `t` is the ramp's own coordinate: 0..1 for the sequential table, -1..1 for
  // the diverging one. Both tables store it in their first column.
  function rampAt(t, table) {
    const lo = table[0][0];
    const hi = table[table.length - 1][0];
    t = Math.max(lo, Math.min(hi, t));
    for (let i = 1; i < table.length; i += 1) {
      if (t <= table[i][0]) {
        const a = table[i - 1];
        const b = table[i];
        const k = (t - a[0]) / (b[0] - a[0]);
        return [a[1] + (b[1] - a[1]) * k,
                a[2] + (b[2] - a[2]) * k,
                a[3] + (b[3] - a[3]) * k];
      }
    }
    const last = table[table.length - 1];
    return [last[1], last[2], last[3]];
  }

  // -- the volume texture -------------------------------------------------
  // The payload is the solver's own (nz, ny, nx, 3) float16 vector field, so
  // turning it into the chosen scalar is one pass over it in components.js --
  // the sample's physical position is reconstructed here because this is the
  // only place that knows both axis orders (the array is z-major, the grid
  // metadata is xyz).
  function positionsOf(grid) {
    const count = grid.nx * grid.ny * grid.nz;
    const positions = new Float32Array(count * 3);
    let i = 0;
    for (let z = 0; z < grid.nz; z += 1) {
      const pz = grid.origin[2] + grid.spacing[2] * z;
      for (let y = 0; y < grid.ny; y += 1) {
        const py = grid.origin[1] + grid.spacing[1] * y;
        for (let x = 0; x < grid.nx; x += 1) {
          positions[i] = grid.origin[0] + grid.spacing[0] * x;
          positions[i + 1] = py;
          positions[i + 2] = pz;
          i += 3;
        }
      }
    }
    return positions;
  }

  function buildTexture(mode) {
    const grid = manifest.grid;
    const buffer = mode.fields;
    if (!buffer || !buffer.byteLength) return null;
    // The share is measured here, off the same decoded vector the scalar volume
    // is built from, so the caption can never describe a different field from
    // the one on screen.
    volumeShare = null;
    const count = grid.nx * grid.ny * grid.nz;
    const expected = count * 3 * 2;   // three components, two bytes each
    if (buffer.byteLength !== expected) {
      // The route sends little-endian half floats, so the length is what says
      // the index and the blob agree. Checking the *loaded* length rather than
      // the decoded one is deliberate: decoding first and comparing afterwards
      // would let a float32 payload -- twice the bytes -- decode into a
      // correctly-shaped array of nonsense and draw a plausible wrong picture.
      throw new Error(
        `field payload is ${buffer.byteLength} bytes, expected ${expected}`);
    }
    const view = new DataView(buffer);
    const vec = new Float32Array(count * 3);
    for (let i = 0; i < vec.length; i += 1) {
      vec[i] = halfToFloat32(view.getUint16(i * 2, true));
    }
    const positions = positionsOf(grid);
    const centre = boxCentre(grid);
    const scalar = componentVolume(vec, positions, centre, system, name);
    volumeShare = componentShare(vec, positions, centre, system, name);

    // Normalise the chosen component by its own peak, so the shader's "1.0 is
    // the strongest sample in this picture" is true.
    //
    // It was not, and the picture was wrong in a way that is easy to miss
    // because it depends on the *mode*: the solver normalises a mode by
    // max|E| over the vector, so |E| always peaks at 1 but an individual
    // component need not. Measured on the live job, mode 1's Erho peaks at
    // 0.4236 and mode 2's at 0.9936, and the shader's threshold is an absolute
    // 0.02 -- so mode 1 discarded 63% of Erho as "no field here" and mode 2
    // discarded 16%, drawing the *same component* at visibly different densities
    // because of nothing but the mode's amplitude. The faint outer lobes that a
    // smoke plot exists to show were the first thing to go.
    //
    // Dividing by the peak is not cosmetic. It makes the threshold a contrast
    // floor rather than an amplitude floor ("draw anything within 50x of this
    // mode's strongest point"), it makes the diverging ramp's mid-band mean the
    // same thing in every mode, and it is what the shader has been claiming in
    // its own comments all along.
    //
    // NaN is skipped, not treated as zero: it marks outside-the-object and must
    // stay NaN so the upload keeps it transparent. A field with no finite
    // non-zero sample (all NaN, or all zero) has no peak to divide by and is
    // left as it is -- every sample is already below any threshold, so the
    // picture is correctly empty.
    let peak = 0;
    for (let i = 0; i < scalar.length; i += 1) {
      const abs = Math.abs(scalar[i]);
      if (abs > peak) peak = abs;
    }
    if (peak > 0 && Number.isFinite(peak)) {
      for (let i = 0; i < scalar.length; i += 1) scalar[i] /= peak;
    }

    // The texture is R32F rather than RGBA16F: the transform is exact in
    // float32 and only the result is narrowed, so a float16 upload would round
    // twice. The values are the component's own, scaled by the peak above so
    // they arrive in -1..1, and NaN is preserved as the sign-and-magnitude half
    // pattern by the narrowing itself.
    const data = new Uint16Array(scalar.length);
    for (let i = 0; i < scalar.length; i += 1) {
      data[i] = float32ToHalf(scalar[i]);
    }
    const texture = new THREE.Data3DTexture(data, grid.nx, grid.ny, grid.nz);
    texture.format = THREE.RedFormat;
    texture.type = THREE.HalfFloatType;
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.wrapR = THREE.ClampToEdgeWrapping;
    texture.unpackAlignment = 1;
    texture.needsUpdate = true;
    return texture;
  }

  function paintModeFailure() {
    const panel = document.getElementById('failure');
    const text = document.getElementById('failure-text');
    if (panel) panel.hidden = false;
    if (text) text.textContent = 'This mode could not be loaded. Try another.';
  }

  function refreshVolume() {
    const mode = modes[current];
    if (!mode) return;
    let next = null;
    try {
      next = buildTexture(mode);
    } catch (err) {
      paintModeFailure();
      return;
    }
    if (!next) { volume.visible = false; return; }
    if (texture) texture.dispose();
    texture = next;
    volumeMaterial.uniforms.uField.value = texture;
    // A component is signed and |E| is not; the shader needs to know which
    // picture it is drawing, because the negative half of a signed component
    // gets the cool end of the ramp rather than being folded onto the warm end.
    volumeMaterial.uniforms.uSigned.value = (name !== 'E');
    volume.visible = true;
    paintLegend();
  }

  // -- mode selection -----------------------------------------------------
  function selectMode(i) {
    current = Math.max(0, Math.min(modes.length - 1, i));
    document.querySelectorAll('#modes-body tr').forEach((tr, n) => {
      tr.classList.toggle('active', n === current);
    });
    const mode = modes[current];
    document.getElementById('mode-caption').textContent = mode
      ? `Mode ${mode.index}: ${formatFrequency(mode.freq_hz)} · ` +
        `${formatWavelength(mode.vacuum_wavelength_m)} in vacuum`
      : '';
    paintDownload(mode);
    loadMode(current);
    refreshVolume();
  }

  function paintDownload(mode) {
    const link = document.getElementById('download-vtu');
    if (!link) return;
    if (!mode || mode.available === false) { link.hidden = true; return; }
    link.hidden = false;
    link.href = `/jobs/${jobId}/files/${vtuName(mode.index)}`;
  }

  // The solver writes `mode_NN.vtu` with two digits, so mode 3 is `mode_03.vtu`.
  // Built from the padded index rather than by listing a directory: the link is
  // a plain file proxied through the app, and guessing a name is what would
  // turn a typo into a 404 the user cannot diagnose.
  function vtuName(index) {
    return `mode_${String(index).padStart(2, '0')}.vtu`;
  }

  // The field for a mode arrives on demand, once. `mode.fields` is the cache and
  // `mode.loading` is the in-flight promise, so selecting a mode twice (or
  // selecting it while it is still downloading) fetches it once. A mode that
  // arrives after the user has moved on still lands in its own cache slot,
  // which is why the next visit is instant. The decision itself lives in
  // modecache.js, where it is tested.
  let inFlight = 0;

  function loadMode(i) {
    const mode = modes[i];
    if (!wantsFetch(i, mode, inFlight)) return;
    inFlight += 1;
    mode.loading = fetch(modeFieldUrl(jobId, mode.index), { cache: 'no-store' })
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(r.status))))
      .then((buffer) => {
        storeFields(mode, buffer);
        if (modes[current] === mode) refreshVolume();
      })
      .catch(() => {
        mode.loading = null;
        mode.failed = true;
        paintModeFailure();
      })
      .then(() => { inFlight -= 1; });
  }

  // -- loop ---------------------------------------------------------------
  function renderLoop() {
    requestAnimationFrame(renderLoop);
    controls.update();
    renderer.render(scene, camera);
  }

  function resize() {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);

  buildSelectors();

  fetch(`/jobs/${jobId}/index.json`, { cache: 'no-store' })
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.status))))
    .then((data) => {
      manifest = data;
      modes = data.modes || [];
      if (!modes.length) return;
      bounds = boundsOf(data.grid);
      fitVolume();
      frameCamera();
      addSurface(jobId);
      selectModeRef = selectMode;
      // Default to mode 1, by the server's own numbering rather than by
      // position -- they coincide today (the solver numbers from 1 and lists in
      // order), but selecting "the mode called 1" is what the owner asked for
      // and stays right if the list ever omits one. Falls back to the first
      // row so an unexpected numbering still shows something.
      const first = modes.findIndex((m) => Number(m.index) === 1);
      selectMode(first === -1 ? 0 : first);
      resize();
      renderLoop();
    })
    .catch(() => {
      document.getElementById('failure').hidden = false;
      document.getElementById('failure-text').textContent =
        'The results could not be loaded. Please refresh the page.';
    });
}

// --------------------------------------------------------------------------
// formatting (also exercised by the tests through a small DOM shim)
// --------------------------------------------------------------------------
export function formatFrequency(hz) {
  if (hz === null || hz === undefined || !Number.isFinite(hz)) return '—';
  if (hz >= 1e12) return `${(hz / 1e12).toFixed(3)} THz`;
  if (hz >= 1e9) return `${(hz / 1e9).toFixed(3)} GHz`;
  if (hz >= 1e6) return `${(hz / 1e6).toFixed(3)} MHz`;
  return `${hz.toExponential(3)} Hz`;
}

export function formatWavelength(m) {
  if (m === null || m === undefined || !Number.isFinite(m)) return '—';
  if (m >= 1e-3) return `${(m * 1e3).toFixed(3)} mm`;
  if (m >= 1e-6) return `${(m * 1e6).toFixed(3)} µm`;
  return `${(m * 1e9).toFixed(3)} nm`;
}

function formatSeconds(s) {
  if (s < 60) return `${s.toFixed(0)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s - m * 60)}s`;
}

function buildModeTable(list) {
  const body = document.getElementById('modes-body');
  if (!body) return;
  body.innerHTML = '';
  list.forEach((mode, i) => {
    const tr = document.createElement('tr');
    tr.dataset.index = String(mode.index);
    tr.innerHTML = `<td>${mode.index}</td><td>${formatFrequency(mode.freq_hz)}</td>` +
                   `<td>${formatWavelength(mode.vacuum_wavelength_m)}</td>`;
    // A mode the index says is unreadable has no field to fetch, so it is not
    // selectable -- clicking it would only produce a failure the user cannot
    // act on. It stays visible so the numbering matches the physics.
    if (mode.available === false) {
      tr.classList.add('unavailable');
      tr.title = 'This mode’s field data is not available.';
    } else {
      // The owner asked for modes to be selectable from the list. A click is
      // the obvious affordance; the keyboard path is what makes it usable
      // without a mouse, so the row is focusable and Enter/Space select it.
      // `selectModeRef` is null until the first payload lands, so guarding the
      // call keeps a stray keypress from throwing before the viewer is ready.
      const select = () => selectModeRef?.(i);
      tr.tabIndex = 0;
      tr.setAttribute('role', 'button');
      tr.addEventListener('click', select);
      tr.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();   // Space would otherwise scroll the page
          select();
        }
      });
    }
    body.appendChild(tr);
  });
}
