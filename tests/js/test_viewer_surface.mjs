// The scene-graph half of the surface loader: which parts of an uploaded glb
// end up in the scene, and whether a load that fails is reported or swallowed.
// Run with:
//   node --test tests/js/*.mjs        <-- the glob matters; see below
//
// DO NOT run `node --test tests/js/`. That directory form discovers ZERO tests,
// prints "pass 0", and EXITS 0 -- so it looks green while having checked
// nothing. Read the COUNT, not the exit code.
//
// WHY THIS FILE EXISTS. The lead's diagnosis listed, as SUSPECTED and not
// verified: "addSurface calls scene.add(child) inside gltf.scene.traverse
// (line ~592-597). Moving a child during traversal can skip siblings, so a
// multi-part model may lose part outlines. Prove or disprove with a 2-mesh
// glb." This file is that proof. The suspicion reproduces, and HARDER than
// stated: with two or more meshes, re-parenting during the walk does not merely
// skip a sibling, it THROWS.
//
// THE MECHANISM, read out of the vendored source rather than inferred. In
// web/static/vendor/three.module.min.js (line 6, char ~69535):
//
//     traverse(t){ t(this); const e=this.children;
//                  for(let n=0,i=e.length;n<i;n++) e[n].traverse(t) }
//
// `e` is the LIVE children array, not a copy, and `i` is its length read once.
// `scene.add(child)` re-parents the child, which splices it out of `e`, so `e`
// shrinks under a loop whose bound `i` does not. The final iteration indexes a
// slot that no longer exists, `e[n]` is `undefined`, and the call throws
// "Cannot read properties of undefined (reading 'traverse')" -- the exact error
// this file was written to chase. A single-child root never reaches the hole
// (the loop ends first), which is why this went unnoticed until a multi-part
// model was uploaded.
//
// So an earlier version of this header -- which claimed traverse "snapshots its
// children" and that the skip-siblings suspicion was disproved -- was WRONG. It
// was a mechanism asserted rather than measured, which this repo has a memory
// for. The first test below now pins the measured mechanism: it asserts the
// throw, at the shape (2 meshes) where the bug is live.
//
// The fix is to collect while traversing and add AFTER the walk, so nothing is
// spliced out of the array being iterated; that also leaves the loader holding
// its own array of parts, which a later re-load or a per-part colour needs. The
// loader does that, and the second test holds it to it.
//
// The real three.js is imported rather than stubbed. The claim under test is
// about three.js's own traverse/add semantics, and a hand-written stub would
// encode my belief about those semantics instead of testing them -- which is
// the "a test that never calls the code cannot fail" shape this repo has a
// memory for.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import * as THREE from '../../web/static/vendor/three.module.min.js';

const here = dirname(fileURLToPath(import.meta.url));
const FIXTURE = join(here, 'fixtures', 'two_part_surface.glb');

/** A stand-in for a parsed glb: the SAME arrangement GLTFLoader produces. */
function twoPartScene() {
  const root = new THREE.Group();
  root.name = 'gltf_scene';
  const a = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1),
                           new THREE.MeshBasicMaterial());
  a.name = 'part_a';
  const b = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1),
                           new THREE.MeshBasicMaterial());
  b.name = 'part_b';
  root.add(a);
  root.add(b);
  return { root, a, b };
}

function meshNames(node) {
  const out = [];
  node.traverse((c) => { if (c.isMesh) out.push(c.name); });
  return out;
}

test('adding a mesh during traverse THROWS on a multi-part model', () => {
  // THE MECHANISM, reproduced and pinned as a throw. This is the "before" shape
  // and it is kept as a test rather than as prose because the claim is about
  // three.js, and three.js could change it under us (an upstream fix that
  // snapshots children would make this go red -- which is the signal to revisit
  // whether the collect-then-add is still load-bearing, not a nuisance).
  //
  // Why it throws, in one line: traverse iterates the LIVE `this.children`
  // array with a length bound captured before the loop (`const e=this.children;
  // for(let n=0,i=e.length;n<i;n++) e[n].traverse(t)`). scene.add() re-parents
  // the child and splices it out of that array, `e` shrinks by one per add, and
  // the last iteration reads `e[n] === undefined` and calls `.traverse` on it.
  const scene = new THREE.Scene();
  const gltf = { scene: twoPartScene().root };
  assert.throws(() => {
    gltf.scene.traverse((child) => {
      if (child.isMesh) scene.add(child);
    });
  }, /Cannot read properties of undefined \(reading 'traverse'\)/,
    'the add-during-traverse form must fail loudly on a 2-part model; if this ' +
    'is now green, three.js changed its traverse and the loader\'s ' +
    'collect-then-add needs re-justifying');

  // And this is WHY it threw: adding part_a spliced it out of the live array,
  // so the loop's next index (1) was past the end of a now-length-1 array. The
  // child that is still in the root is part_b -- the one never reached, because
  // the walk died on the hole before visiting it. (The throw is at t(this), the
  // first line of the recursive call, so it happens on the undefined child
  // rather than after the loop runs out: exactly one add succeeded.)
  assert.equal(gltf.scene.children.length, 1,
    'one part was re-parented out and the other was never reached: the walk ' +
    'died on the hole the first add left behind');
  assert.deepEqual(meshNames(gltf.scene), ['part_b'],
    'the surviving mesh is the one the walk never reached -- proof that the ' +
    'throw, not a quiet skip, is what loses parts');
});

test('every mesh reaches the scene with the outline material, names kept aside', () => {
  // THE FIX, as the loader now does it: collect while traversing, add after,
  // and keep the parts in the loader's own array.
  //
  // WHAT IS AND IS NOT REQUIRED HERE. The lead ruled on this explicitly: "The
  // requirement is that EVERY mesh ends up in the scene and gets the outline
  // material. The glb hierarchy does not have to survive. ... Keep the part
  // identity in your own array if you need it." So the glb root being emptied
  // is accepted, NOT a defect -- an earlier version of this test asserted the
  // root survived, which was my premise rather than the requirement, and it was
  // wrong: Object3D.add() re-parents unconditionally, so BOTH forms empty it.
  // What the test holds the loader to is the pair that does differ: the parts
  // are all present, and the loader still knows their names.
  const scene = new THREE.Scene();
  const gltf = { scene: twoPartScene().root };
  const outline = new THREE.MeshBasicMaterial();
  const parts = [];
  gltf.scene.traverse((child) => {
    if (child.isMesh) {
      child.material = outline;
      parts.push(child);
    }
  });
  for (const p of parts) scene.add(p);

  assert.equal(parts.length, 2, 'the traversal found both parts');
  assert.deepEqual(meshNames(scene).sort(), ['part_a', 'part_b'],
    'every mesh of a multi-part model is in the scene; a loader that added ' +
    'during the walk and lost one would fail exactly here');

  // The material really landed on every part, not just the one surfaceMesh was
  // last assigned to -- the single-`surfaceMesh` version of this code kept only
  // the last mesh, which is the same defect seen from the material side.
  const got = [];
  scene.traverse((c) => { if (c.isMesh) got.push(c.material); });
  assert.equal(got.length, 2, 'both meshes are in the scene');
  for (const m of got) {
    assert.equal(m, outline, 'every part carries the outline material');
  }

  // Part identity survives in the loader's own array even though the glb root
  // does not. This is the property the lead said to keep if it was needed, and
  // it is what a later re-load or a per-part colour would address.
  assert.deepEqual(parts.map((p) => p.name).sort(), ['part_a', 'part_b'],
    'the collected array keeps every part addressable by name');
});

test('the two-part fixture really does carry two meshes in two nodes', () => {
  // The fixture is what makes the test above meaningful rather than vacuous: a
  // single-mesh glb would satisfy every assertion in the fix test. This reads
  // the checked-in binary and counts, so a fixture regenerated wrong is a loud
  // failure here rather than a silent weakening of the test above.
  assert.ok(existsSync(FIXTURE), `the fixture is missing at ${FIXTURE}`);
  const buf = readFileSync(FIXTURE);
  assert.equal(buf.readUInt32LE(0), 0x46546c67, 'the fixture is not a glb');
  const jsonLen = buf.readUInt32LE(12);
  const json = JSON.parse(buf.subarray(20, 20 + jsonLen).toString('utf8'));
  assert.equal(json.meshes.length, 2, 'the fixture must have two meshes');
  assert.equal(json.nodes.length, 2, 'and two nodes');
  assert.equal(json.scenes.length, 1, 'in one scene');
  assert.deepEqual(json.scenes[0].nodes, [0, 1],
    'the scene must reference both nodes -- a fixture listing one node would ' +
    'make the scene-graph tests pass without ever exercising two parts');
  // And the two parts must be in DIFFERENT meshes, not two primitives of one:
  // the traverse/add bug is about mesh nodes.
  assert.notEqual(json.nodes[0].mesh, json.nodes[1].mesh,
    'both nodes point at the same mesh, so there is only one part after all');
});

// The viewer's OWN source, for the guards below. The scene-graph tests above
// exercise three.js against a stand-in; these check that the loader actually
// wires the surface into the scene at the conversion the grid implies. That is
// a different question -- what the real file does with a real grid -- and a
// test that never reads the real file cannot answer it.
const VIEWER = readFileSync(join(here, '..', '..', 'web', 'static', 'viewer.js'),
                            'utf8');

test('the loader scales the surface into the grid units, and says so when it cannot', () => {
  // THE BUG THIS PINS, in the loader rather than in the geometry module: the
  // surface was drawn in its own units (mm) while the field grid is in metres,
  // so the outline was ~1000x too big and off the origin -- the "cubic region"
  // the owner saw. tests/js/test_surface.mjs checks that the CONVERSION the
  // module computes is right; this checks that the loader APPLIES it, which is
  // the half that was missing when this row arrived (surface.js existed and was
  // not imported by viewer.js at all).
  assert.match(VIEWER, /from\s*'\.\/surface\.js'/,
    'viewer.js does not import surface.js, so the unit conversion is dead code ' +
    'and the outline is drawn in the surface\'s own units -- the original bug');
  assert.match(VIEWER, /surfaceScaleFromGrid\(/,
    'the loader never calls surfaceScaleFromGrid: the surface is drawn unscaled');
  // The scale must actually reach the object's transform, and it must be the
  // COMPUTED scale -- not just any call. A guard that matched the bare
  // `surfaceGroup.scale.set(` call stayed green against `scale.set(1, 1, 1)`,
  // which is the same bug (surface drawn at 1x) wearing the right method name.
  // So match the argument: the call must pass the `scale` array through.
  const applied = /surfaceGroup\.scale\.set\(\s*scale\[0\]\s*,\s*scale\[1\]\s*,\s*scale\[2\]\s*\)/.exec(VIEWER);
  assert.ok(applied,
    'the surface is not scaled by the computed scale: a constant (or a ' +
    'different vector) here draws the outline in the wrong units, which is ' +
    'the original defect -- the call existing is not the property, the ' +
    'argument being the computed scale is');
  // And the "could not determine" case must not fall through to drawing at 1x:
  // a falsy scale has to be handled before the add, not after.
  const guard = /if\s*\(\s*!\s*scale\s*\)[\s\S]{0,400}?return;/.exec(VIEWER);
  assert.ok(guard,
    'a missing scale is not handled with an early return, so an undeterminable ' +
    'conversion would fall through and draw the surface at 1x -- "could not ' +
    'determine" masquerading as "no scaling", which is exactly the bug');
});

test('the loader collects parts before adding them, so it never re-parents in a walk', () => {
  // The scene-graph tests above prove the MECHANISM against a stand-in: with
  // the real three.js, calling scene.add(child) inside gltf.scene.traverse
  // splices the child out of the live children array and throws. But a
  // stand-in cannot hold THIS file to the fix -- the loader could still call
  // add inside its own walk, and those tests would stay green, because they
  // never run the loader. That gap is not hypothetical: reinstating
  // `scene.add(child)` inside viewer.js's traverse leaves the whole suite
  // passing. So the loader's own source is checked here, on the property that
  // matters: the only scene.add inside the walk must not exist, and the add
  // must happen after the walk has finished.
  const walk = /gltf\.scene\.traverse\([\s\S]{0,600}?\}\);/.exec(VIEWER);
  assert.ok(walk, 'the loader no longer walks gltf.scene; the shape changed ' +
                  'and this guard needs re-pointing at the new one');
  assert.ok(!/scene\.add\s*\(/.test(walk[0]),
    'the loader adds to the scene INSIDE gltf.scene.traverse: re-parenting a ' +
    'child during the walk splices it out of the live children array and ' +
    'throws "Cannot read properties of undefined (reading \'traverse\')" on a ' +
    'multi-part model. Collect first, add after -- which is what the walk ' +
    'below is for');
  // And the add must still happen, after the walk, for every collected part.
  // Absence of the bad call is not the fix; the parts have to reach the scene.
  const collected = /const\s+parts\s*=\s*\[\][\s\S]{0,200}?traverse\(/.exec(VIEWER);
  assert.ok(collected,
    'the loader no longer collects into a `parts` array before the walk');
  // `for (const part of parts)` occurs twice in the file, so this must anchor
  // on the one that adds to the group -- the collect-then-add loop. Matching
  // either one would let the loop be emptied while this stayed green.
  const addAfter = /for\s*\(\s*const\s+part\s+of\s+parts\s*\)\s*\{[\s\S]*?surfaceGroup\.add\(\s*part\s*\)/.exec(VIEWER);
  assert.ok(addAfter,
    'the parts are never added after the walk: collect-then-add must end with ' +
    'every part reaching the group, or the outline is silently missing');
  // The guard above reads a window of the file; make sure it is the loader's
  // loop and not some later one, by requiring it to sit after the walk.
  assert.ok(addAfter.index > walk.index,
    'the add-after-the-walk loop was found BEFORE the walk -- the regex has ' +
    'latched onto the wrong `for (const part of parts)` and this guard is ' +
    'checking a loop that is not the collect-then-add one');
});
