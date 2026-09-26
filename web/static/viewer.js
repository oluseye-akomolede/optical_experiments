// MSE mode viewer: the object, coloured field slices, and E-direction arrows.
//
// Data comes from two endpoints, both same-origin:
//   /jobs/<id>/status      small poll for progress
//   /jobs/<id>/slices.json every mode's slices + a coarse point cloud
//   /jobs/<id>/files/...   proxied results (modes.json, surface.glb)
//
// three.js is vendored under /static/vendor (pinned 0.160.0, MIT); nothing is
// fetched from a CDN, because the app is only reachable on the tailnet.

import * as THREE from './vendor/three.module.min.js';
import { GLTFLoader } from './vendor/GLTFLoader.js';

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
    loadFieldData(jobId, data, { modesBody, viewer });
  }

  tick();
}

// --------------------------------------------------------------------------
// Field data + interaction
// --------------------------------------------------------------------------
function loadFieldData(jobId, statusData, dom) {
  const canvas = document.getElementById('scene');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0e12);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);

  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(3, 5, 4);
  scene.add(key);

  const group = new THREE.Group();
  scene.add(group);

  const slicePlanes = {
    x: makePlane(),
    y: makePlane(),
    z: makePlane(),
  };
  for (const plane of Object.values(slicePlanes)) group.add(plane.mesh);

  const arrowGroup = new THREE.Group();
  arrowGroup.visible = false;
  group.add(arrowGroup);

  // the object itself, semi-transparent so the slices read through it
  let surfaceMesh = null;

  let manifest = null;
  let modes = [];
  let current = 0;
  let animate = false;
  let phase = 0;
  let bounds = null;
  let lastTouched = 'z';

  fetch(`/jobs/${jobId}/slices.json`, { cache: 'no-store' })
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.status))))
    .then((data) => {
      manifest = data;
      modes = data.modes || [];
      if (!modes.length) return;
      bounds = boundsOf(data.grid);
      frameCamera();
      addSurface(jobId, bounds);
      buildSliders(data.grid);
      selectModeRef = selectMode;
      selectMode(0);
      resize();
      renderLoop();
    })
    .catch(() => {
      document.getElementById('failure').hidden = false;
      document.getElementById('failure-text').textContent =
        'The results could not be loaded. Please refresh the page.';
    });

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

  function frameCamera() {
    const radius = Math.max(...bounds.size) * 1.6 || 1;
    camera.position.set(bounds.centre[0] + radius,
                        bounds.centre[1] + radius * 0.8,
                        bounds.centre[2] + radius);
    camera.lookAt(new THREE.Vector3(...bounds.centre));
  }

  function addSurface(id, b) {
    fetch(`/jobs/${id}/files/surface.glb`)
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(r.status))))
      .then((buffer) => new Promise((resolve, reject) => {
        new GLTFLoader().parse(buffer, '', resolve, reject);
      }))
      .then((gltf) => {
        const material = new THREE.MeshStandardMaterial({
          color: 0x9fb6cc, metalness: 0.1, roughness: 0.65,
          transparent: true, opacity: 0.22, side: THREE.DoubleSide,
          depthWrite: false,
        });
        gltf.scene.traverse((child) => {
          if (child.isMesh) {
            child.material = material;
            surfaceMesh = child;
            group.add(child);
          }
        });
      })
      .catch(() => { /* the field is the point; a missing surface is survivable */ });
  }

  function makePlane() {
    const geometry = new THREE.PlaneGeometry(1, 1);
    geometry.rotateX(-Math.PI / 2);   // lie flat in the x-z plane by default
    const material = new THREE.MeshBasicMaterial({
      side: THREE.DoubleSide, transparent: true, opacity: 0.92,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geometry, material);
    return { mesh, geometry, material, texture: null };
  }

  function buildSliders(grid) {
    for (const axis of ['x', 'y', 'z']) {
      const input = document.getElementById(`slice-${axis}`);
      const count = { x: grid.nx, y: grid.ny, z: grid.nz }[axis];
      input.max = String(count - 1);
      input.value = String(Math.floor((count - 1) / 2));
      input.addEventListener('input', () => {
        lastTouched = axis;
        updateSlices();
      });
    }
    document.getElementById('toggle-arrows').addEventListener('click', (e) => {
      arrowGroup.visible = !arrowGroup.visible;
      e.target.classList.toggle('secondary', arrowGroup.visible);
      updateSlices();
    });
    document.getElementById('toggle-animate').addEventListener('click', (e) => {
      animate = !animate;
      e.target.classList.toggle('secondary', animate);
    });
  }

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
    updateSlices();
  }

  function updateSlices() {
    const mode = modes[current];
    if (!mode) return;

    for (const axis of ['x', 'y', 'z']) {
      const wanted = Number(document.getElementById(`slice-${axis}`).value);
      const slice = mode.slices.find((s) => s.axis === axis);
      const plane = slicePlanes[axis];
      if (!slice) { plane.mesh.visible = false; continue; }
      plane.mesh.visible = true;
      plane.slice = slice;
      paint(plane, slice);
      placePlane(plane, slice, wanted, axis);
    }
    if (arrowGroup.visible) updateArrows();
    else clearArrows();
  }

  function paint(plane, slice) {
    // Every mode is normalised to max|E| = 1 by the solver, so one colour scale
    // is shared across modes and the list stays comparable.
    const globalMax = slice.mag.max || 1;
    const uN = slice.u.count;
    const vN = slice.v.count;
    const canvas = document.createElement('canvas');
    canvas.width = uN;
    canvas.height = vN;
    const ctx = canvas.getContext('2d');
    const image = ctx.createImageData(uN, vN);
    const mag = slice.mag.data;
    for (let v = 0; v < vN; v += 1) {
      for (let u = 0; u < uN; u += 1) {
        const value = mag[v * uN + u];
        const [r, g, b, a] = value === null
          ? [0, 0, 0, 0]
          : colormap(value / (globalMax || 1));
        const o = (v * uN + u) * 4;
        image.data[o] = r; image.data[o + 1] = g;
        image.data[o + 2] = b; image.data[o + 3] = a;
      }
    }
    ctx.putImageData(image, 0, 0);
    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    if (plane.texture) plane.texture.dispose();
    plane.texture = texture;
    plane.material.map = texture;
    plane.material.needsUpdate = true;
  }

  // A node-scale perceptual ramp: deep blue -> cyan -> green -> yellow -> red.
  function colormap(t) {
    const x = Math.max(0, Math.min(1, t));
    const stops = [
      [0.00, 0x0d, 0x2c, 0x6b],
      [0.35, 0x1f, 0x8a, 0xd6],
      [0.60, 0x43, 0xd1, 0x7c],
      [0.82, 0xff, 0xc8, 0x57],
      [1.00, 0xff, 0x5a, 0x3c],
    ];
    for (let i = 1; i < stops.length; i += 1) {
      if (x <= stops[i][0]) {
        const [t0, r0, g0, b0] = stops[i - 1];
        const [t1, r1, g1, b1] = stops[i];
        const k = (x - t0) / (t1 - t0);
        return [Math.round(r0 + (r1 - r0) * k),
                Math.round(g0 + (g1 - g0) * k),
                Math.round(b0 + (b1 - b0) * k), 235];
      }
    }
    return [255, 90, 60, 235];
  }

  // The plane geometry starts flat on the ground (spans x and z), so:
  //   normal y -> no rotation
  //   normal z -> rotate about x to stand it up
  //   normal x -> rotate about z to stand it up and face x
  // `slice.u` is the horizontal in-plane axis and `slice.v` the vertical one,
  // and their names tell us which MODEL axis each one runs along.
  function placePlane(plane, slice, wanted, axis) {
    const w = Math.abs(slice.u.spacing_m) * Math.max(slice.u.count - 1, 1);
    const h = Math.abs(slice.v.spacing_m) * Math.max(slice.v.count - 1, 1);
    const u0 = slice.u.origin_m;
    const v0 = slice.v.origin_m;
    const at = slicePosition(slice, wanted);

    if (axis === 'y') {
      // u = x, v = z, normal = y
      plane.mesh.rotation.set(0, 0, 0);
      plane.mesh.scale.set(w || 1e-6, 1, h || 1e-6);
      plane.mesh.position.set(u0 + w / 2, at, v0 + h / 2);
    } else if (axis === 'z') {
      // u = x, v = y, normal = z
      plane.mesh.rotation.set(Math.PI / 2, 0, 0);
      plane.mesh.scale.set(w || 1e-6, 1, h || 1e-6);
      plane.mesh.position.set(u0 + w / 2, v0 + h / 2, at);
    } else {
      // u = y, v = z, normal = x
      plane.mesh.rotation.set(0, 0, Math.PI / 2);
      plane.mesh.scale.set(h || 1e-6, 1, w || 1e-6);
      plane.mesh.position.set(at, u0 + w / 2, v0 + h / 2);
    }
  }

  function clearArrows() {
    while (arrowGroup.children.length) {
      const arrow = arrowGroup.children.pop();
      arrowGroup.remove(arrow);
      if (arrow.line) { arrow.line.geometry.dispose(); arrow.line.material.dispose(); }
    }
  }

  function updateArrows() {
    clearArrows();
    const axis = activeAxis();
    const slice = modes[current].slices.find((s) => s.axis === axis);
    if (!slice) return;
    // The slider value is an INDEX along the normal axis; where that plane sits
    // in metres is `wantedPos`, which is also the third coordinate of every
    // arrow on it.
    const wanted = Number(document.getElementById(`slice-${axis}`).value);
    const stride = Math.max(1, Math.round(slice.u.count / 14),
                              Math.round(slice.v.count / 14));
    const uN = slice.u.count;
    const vec = slice.vec.data;
    const mag = slice.mag.data;
    const peak = slice.mag.max || 1;
    const length = Math.max(0.02, Math.max(...bounds.size) * 0.06);

    for (let v = 0; v < slice.v.count; v += stride) {
      for (let u = 0; u < uN; u += stride) {
        const n = v * uN + u;
        if (mag[n] === null) continue;
        const strength = mag[n] / peak;
        if (strength < 0.05) continue;
        const [ex, ey, ez] = [vec[n * 3], vec[n * 3 + 1], vec[n * 3 + 2]];
        const dir = new THREE.Vector3(ex, ey, ez);
        if (dir.length() < 1e-12) continue;
        // array order is (z, y, x) for the plane, matching the grid axes; the
        // vector components are already in model x/y/z
        const p = pointFor(slice, wanted, u, v);
        const osc = animate ? Math.cos(phase) : 1;
        const arrow = new THREE.ArrowHelper(
          dir.clone().normalize(), p, length * strength * Math.abs(osc),
          colourFor(strength), length * 0.4, length * 0.22);
        arrowGroup.add(arrow);
      }
    }
  }

  function pointFor(slice, wanted, u, v) {
    // The slice's own u/v axes live in the plane; the third coordinate is the
    // plane's position along its normal.
    const uPos = slice.u.origin_m + slice.u.spacing_m * u;
    const vPos = slice.v.origin_m + slice.v.spacing_m * v;
    const at = slicePosition(slice, wanted);
    if (slice.axis === 'x') return new THREE.Vector3(at, uPos, vPos);
    if (slice.axis === 'y') return new THREE.Vector3(uPos, vPos, at);
    return new THREE.Vector3(vPos, uPos, at);
  }

  // Where the plane sits, in metres, for a slider index. x/y/z here are the
  // MODEL axes; `slice.u.name`/`slice.v.name` say which of them each in-plane
  // axis actually is.
  function slicePosition(slice, index) {
    const grid = manifest.grid;
    const axisIndex = { x: 0, y: 1, z: 2 }[slice.axis];
    return grid.origin[axisIndex] + grid.spacing[axisIndex] * index;
  }

  function colourFor(strength) {
    const [r, g, b] = colormap(strength);
    return new THREE.Color(r / 255, g / 255, b / 255);
  }

  // Which plane the arrows are drawn on: the one whose slider moved last.
  function activeAxis() {
    return lastTouched;
  }

  // -- animation ----------------------------------------------------------
  function renderLoop() {
    requestAnimationFrame(renderLoop);
    if (animate) {
      phase += 0.08;
      if (arrowGroup.visible) updateArrows();
    }
    if (surfaceMesh) surfaceMesh.rotation.y = 0;   // reserved for later interactivity
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
    tr.addEventListener('click', () => selectModeRef?.(i));
    body.appendChild(tr);
  });
}
