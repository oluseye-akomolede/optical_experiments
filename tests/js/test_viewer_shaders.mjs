// The viewer's shader source, checked as *source*.
//
// Why this file exists: the shaders are JavaScript template literals, and a
// backtick inside one ends the string early. The result is a file that fails to
// parse in the browser -- the page shows nothing, `#viewer` never appears, and
// the only symptom is a console line reading "Unexpected identifier 's'".
//
// That happened twice while writing the volume plot's comments, and it is
// genuinely hard to see: the comment reads as prose, the backticks look like
// ordinary emphasis, and the error names a token (an 's') that appears nowhere
// near the offending line. Worse, `node --check` passes when the stray backticks
// come in pairs, because the parse still succeeds -- it just parses
// *differently*, silently reinterpreting the rest of the file. So the parse
// check that is normally enough for a JS file is not enough here, and the check
// has to be aimed at the literal's contents.
//
// This is the cheap structural version. It reads the shader literals out of the
// source and asserts none of them contains a backtick, which is the property
// that was violated. The render check is the expensive behavioural version and
// would also catch it -- but only by timing out on a missing selector, which
// says nothing about why. This one says why.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const source = readFileSync(join(here, '..', '..', 'web', 'static', 'viewer.js'),
                            'utf8');

/**
 * Pull out the body of every `const NAME = \`...\`;` at column 0.
 *
 * Deliberately a small hand-rolled scan rather than a real JS parse: the point
 * is to inspect the *text between the delimiters*, which a parser would consume
 * and a linter would not report. It only looks at top-level (column-0) consts,
 * which is where the shaders are declared.
 */
function shaderLiterals(text) {
  const out = [];
  const lines = text.split('\n');
  let name = null;
  let body = null;
  for (const line of lines) {
    if (body === null) {
      const m = /^const ([A-Z_][A-Z0-9_]*) = `(.*)$/.exec(line);
      if (m) {
        name = m[1];
        body = [m[2]];
        // a one-line literal closes on the same line
        if (m[2].includes('`')) {
          out.push({ name, body: m[2].slice(0, m[2].indexOf('`')) });
          name = null;
          body = null;
        }
      }
      continue;
    }
    if (line === '`;') {
      out.push({ name, body: body.join('\n') });
      name = null;
      body = null;
      continue;
    }
    body.push(line);
  }
  return out;
}

test('the shader literals contain no stray backticks', () => {
  const shaders = shaderLiterals(source);
  // A positive control: if the scan found nothing, it is broken, not the file.
  // Without this the test would pass on a viewer whose shaders it never read.
  assert.ok(shaders.length >= 2,
    `expected to find the shader literals, found ${shaders.length}: ` +
    `${shaders.map((s) => s.name).join(', ')}`);

  for (const { name, body } of shaders) {
    // The body is what a naive scan captured. A stray backtick means the scan
    // stopped early, so the body is shorter than the real literal -- which is
    // exactly the failure being caught, and it shows up here as a backtick.
    assert.ok(!body.includes('`'),
      `${name} contains a backtick, which would end the template literal ` +
      'early and break the page');
  }
});

test('the two ramps are both present and are different', () => {
  // The signed and magnitude plots must not share a ramp: the defect that
  // prompted this was one sequential ramp being used for both, with the sign
  // applied by multiplying the colour by -1 (which clamps to black).
  assert.match(source, /colormapSequential/,
    'the sequential ramp for |E| is missing');
  assert.match(source, /colormapDiverging/,
    'the diverging ramp for signed components is missing');
  // The broken spelling specifically: never multiply a colour by a sign.
  // `colormap\w*` rather than an exact name, because the first version of this
  // check matched only `colormap(...)` and so missed the mutation that renamed
  // the call to `colormapSequential(...) * sign(v)` -- a guard that passes on
  // the bug it exists for is worse than no guard, since it is trusted.
  assert.ok(!/colormap\w*\([^)]*\)\s*\*\s*sign\(/.test(source),
    'a colour is being multiplied by sign(v) again -- that clamps negative ' +
    'RGB to black, which is what drew the black half-disc');
});

test('the component is normalised by its own peak before upload', () => {
  // The bug this pins: the shader thresholds an absolute `t > uThreshold` and
  // the diverging ramp is documented as taking 1.0 = the strongest sample, but
  // the build uploaded the component raw. A mode is normalised by max|E| over
  // the *vector*, so |E| peaks at 1 while a component need not -- measured on
  // the live job, mode 1's Erho peaks at 0.4236 and mode 2's at 0.9936. The
  // absolute threshold therefore threw away 63% of mode 1's Erho and 16% of
  // mode 2's, so the same component drew at different densities for no reason
  // but the mode's amplitude, and faint outer lobes vanished.
  //
  // The assertion is that a divide by a peak exists between building the
  // scalar and narrowing it to halves. It is deliberately about the *wiring*
  // rather than the arithmetic: the peak must be computed from the scalar and
  // must be applied before the float16 narrow, or it is normalising something
  // the shader never sees.
  const build = /const scalar = componentVolume\(([^;]*)\);([\s\S]*?)const data = new Uint16Array/.exec(source);
  assert.ok(build, 'the texture build has changed shape -- this guard needs rewriting');
  assert.match(build[2], /peak/,
    'the component is no longer normalised by its own peak before upload: an ' +
    'absolute threshold on an unnormalised component silently discards the ' +
    'faint structure of weak modes');
  assert.match(build[2], /scalar\[i\]\s*\/=\s*peak|scalar\[i\]\s*=\s*scalar\[i\]\s*\/\s*peak/,
    'a peak is computed but never divided into the scalar -- the shader still ' +
    'receives the raw values');
});

test('the raymarch step is a fraction of the box, not a physical length', () => {
  // The opacity bug: multiplying the per-step alpha by the box's physical size
  // made the plot's visibility depend on how big the object was, and on the
  // live job it came out as 4% opacity -- invisible.
  //
  // This asserts the whole alpha expression, not a substring of it. Matching
  // `uAlphaScale * dt` and forbidding `uExtent` would let `* 10.0` through, and
  // a hard-coded constant is the same class of defect: an alpha that is right
  // for one grid and wrong for another. Anything between `clamp(` and the
  // closing paren that is not exactly `t * uAlphaScale * dt` is a change to the
  // optical model and should have to come here and say so.
  const alpha = /float a = clamp\(([^;]*?), 0\.0, 1\.0\)/.exec(source);
  assert.ok(alpha, 'the alpha integral has gone or changed shape');
  const expression = alpha[1].replace(/\s+/g, ' ').trim();
  assert.equal(expression, 't * uAlphaScale * dt',
    `the alpha integral is "${expression}", not "t * uAlphaScale * dt" -- ` +
    'any physical length or hard-coded scale here makes the plot\'s opacity ' +
    'depend on the grid rather than on the field');
});

test('a sample at the box face fetches the last texel, not one past it', () => {
  // The NaN fallback is `texelFetch(clamp(ivec3(floor(uvw * n)), 0, n-1))`.
  // At uvw == 1.0 the raw index would be n, one past the last texel, and an
  // out-of-range texelFetch is undefined in WebGL2.
  //
  // What this test is NOT for, and used to claim wrongly: it does not close an
  // observable defect. The raymarch reads sample i at offset (i + jitter) * dt
  // with i <= steps-1 and jitter < 1, so the largest offset read is strictly
  // less than span and uvw never reaches 1.0 -- measured, the live grid's
  // maximum is 0.999999994. A sweep that drives sampleField at uvw == 1.0
  // directly reports a large out-of-range count and means nothing, because the
  // march never produces that coordinate; that mistake was made here, reported
  // as 481201 samples, and retracted once the march was replayed instead of
  // swept. See the comment in viewer.js beside the clamp.
  //
  // It is still worth a guard: the clamp's bound must stay "the last texel". A
  // future edit could replace n - 1 with a constant that happens to suit one
  // grid, or drop the clamp, and nothing on screen would fail on a driver that
  // zeroes out-of-range fetches.
  // Match from `return sampleTexel(` to the terminating `);` on its own, with a
  // greedy run -- a lazy or paren-excluding pattern stops inside the nested
  // ivec3(...) and captures a truncated expression, which is how this guard
  // first failed against the very clamp it exists to check.
  const body = source.slice(source.indexOf('if (bad || isNan(v000)'),
                            source.indexOf('float x00'));
  const fallback = /return sampleTexel\((.*)\);/s.exec(body);
  assert.ok(fallback && /clamp\(/.test(fallback[1]),
    'the NaN fallback texelFetch is no longer clamped: at uvw == 1.0 ' +
    'floor(uvw * n) is n, one past the last texel, and an out-of-range ' +
    'texelFetch is undefined in WebGL2');

  // The upper bound is read OUT OF THE SHADER, not re-typed here. This is the
  // difference between a guard and a re-implementation: an earlier version of
  // this test hard-coded the expected bound, and mutation testing showed it let
  // both `clamp(c, ivec3(0), ivec3(63))` and `clamp(c, ivec3(0), n - ivec3(2))`
  // through -- the test was checking its own arithmetic, not the shader's. So
  // the bound expression is extracted from the source and evaluated against the
  // real grid, and the assertion is that it equals n-1 exactly. A bound that is
  // a constant, or off by one in either direction, now fails.
  // The upper bound is the clamp's THIRD argument, split by balancing parens
  // rather than by regex: the bound itself contains parens (`ivec3(1)`), and
  // every regex shape tried here captured it one character short -- the outer
  // `clamp(` and the inner `ivec3(` close in the same place, so counting parens
  // is the only thing that gets it right.
  function splitArgs(expr) {
    const out = [];
    let depth = 0;
    let cur = '';
    for (const ch of expr) {
      if (ch === '(') depth += 1;
      if (ch === ')') depth -= 1;
      if (ch === ',' && depth === 0) { out.push(cur.trim()); cur = ''; continue; }
      cur += ch;
    }
    if (cur.trim()) out.push(cur.trim());
    return out;
  }
  const clampCall = fallback[1].slice(fallback[1].indexOf('(') + 1,
                                      fallback[1].lastIndexOf(')'));
  const callArgs = splitArgs(clampCall);
  assert.equal(callArgs.length, 3,
    `the fallback clamp has ${callArgs.length} arguments, not 3 -- this guard ` +
    'needs rewriting against the new shape');
  const boundText = callArgs[2];
  // Only the forms the shader can legitimately use are allowed through; a bound
  // that is a hard-coded integer is rejected by name rather than evaluated, so
  // `ivec3(63)` cannot pass on a grid whose last texel happens to be 63.
  if (!/\bn\b/.test(boundText)) {
    throw new Error(`the clamp's upper bound "${boundText}" does not use the ` +
      'texture size n, so it is right for one grid and wrong for the next');
  }
  // Evaluate the shader's own bound expression. Only the spellings the shader
  // can legitimately use are recognised; anything else THROWS rather than
  // evaluating to something that happens to compare equal, so a bound this
  // guard does not understand is a loud failure, not a silent pass. In
  // particular `ivec3(63)` has no `n` and was already rejected above, but it is
  // also simply not one of the shapes below.
  const VEC = (a) => [a, a, a];
  const SUB = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const upper = (n) => {
    const t = boundText.replace(/\s+/g, '');
    if (t === 'n') return n.slice();
    if (t === 'n-ivec3(1)') return SUB(n, VEC(1));
    if (t === 'ivec3(n)-ivec3(1)') return SUB(VEC(n), VEC(1));
    throw new Error(`the fallback clamp's upper bound is "${boundText}", which ` +
      'this guard does not recognise as "the last texel" -- if it is a new ' +
      'legitimate spelling, teach the guard; if it is a constant or an ' +
      'off-by-one, it is the bug this test exists for');
  };

  const n = [64, 45, 32];               // the live job's grid
  assert.deepEqual(upper(n), [63, 44, 31],
    `the fallback clamp's upper bound evaluates to ${JSON.stringify(upper(n))} ` +
    `for grid ${JSON.stringify(n)}, not the last texel ${JSON.stringify(n.map((v) => v - 1))}`);

  const clamped = (uvw) => [0, 1, 2].map((a) =>
    Math.min(Math.max(Math.floor(uvw[a] * n[a]), 0), upper(n)[a]));

  for (const uvw of [[1, 1, 1], [1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1],
                     [0, 0, 0], [0.5, 0.5, 0.5]]) {
    const c = clamped(uvw);
    for (let a = 0; a < 3; a += 1) {
      assert.ok(c[a] >= 0 && c[a] < n[a],
        `a fallback sample at uvw=${JSON.stringify(uvw)} fetches texel ` +
        `index ${c[a]} on axis ${a}, outside 0..${n[a] - 1} -- the clamp is ` +
        'not holding, and an out-of-range texelFetch is undefined');
    }
  }

  // The two faces specifically: the high face must land ON the last texel (not
  // short of it -- a clamp that clamps too much samples the wrong voxel) and
  // the low face on the first.
  assert.deepEqual(clamped([1, 1, 1]), [63, 44, 31],
    'uvw == 1.0 must fetch the last texel of each axis, not n and not n-2');
  assert.deepEqual(clamped([0, 0, 0]), [0, 0, 0],
    'uvw == 0.0 must fetch the first texel');
});

test('the faint-component caption is driven by the measured share, not the name', () => {
  // The render check asserts the caption's *text* against a live page; this
  // asserts the wiring that produces it, which is the part a live page cannot
  // distinguish. Two ways the caption goes wrong and both look fine on screen:
  //
  //  1. It never fires, so a faint component is again indistinguishable from a
  //     broken one -- the whole reason the caption exists.
  //  2. It fires off the component's *name* rather than the measured share, so
  //     it calls Erho small in every mode and tells the user a lie about mode 2
  //     (where Erho reaches 98% of |E|).
  assert.match(source, /volumeShare\s*=\s*componentShare\(/,
    'the viewer no longer measures the component share, so any caption must be ' +
    'guessing from the component name alone');

  const note = /function paintComponentNote\([^)]*\)\s*\{([\s\S]*?)\n  \}/.exec(source);
  assert.ok(note, 'paintComponentNote has gone -- the faint-component caption ' +
                  'is no longer rendered at all');
  assert.match(note[1], /volumeShare\s*<\s*SMALL_SHARE/,
    'the caption no longer compares the measured share against a threshold, so ' +
    'it cannot be tracking the field');
  // The unmeasurable branch. This assertion started out checking only that the
  // `volumeShare === null` CONDITION survived, and mutation testing showed why
  // that is not enough: replacing the branch's whole body -- the honest "Could
  // not measure" message -- with `volumeShare = 1;` left the condition in place
  // and the guard passed. A guard satisfied by the test in an if-statement
  // says nothing about what the branch does, and the branch is where the
  // could-not-ask honesty lives. So the branch body is now required to produce
  // a message that says it could not measure.
  const unmeasurable =
    /if\s*\(\s*volumeShare === null \|\| !Number\.isFinite\(volumeShare\)\s*\)\s*\{([\s\S]*?)\n    \}/
      .exec(note[1]);
  assert.ok(unmeasurable,
    'the caption no longer has an unmeasurable-share branch, so a share that ' +
    'could not be measured is reported as though it had been');
  assert.match(unmeasurable[1], /Could not measure/i,
    'the unmeasurable-share branch no longer says it could not measure -- a ' +
    'caption that reads like a measurement when it is not one is the ' +
    'could-not-ask defect wearing a caption\'s clothes');
  assert.match(unmeasurable[1], /return\s*;/,
    'the unmeasurable-share branch must return, not fall through and print a ' +
    'share it never measured');

  // And it must be reachable: a caption nothing calls is the same as no caption.
  assert.match(source, /paintComponentNote\(/,
    'paintComponentNote is defined but never called');
  const legend = /function paintLegend\(\)\s*\{([\s\S]*?)\n  \}/.exec(source);
  assert.ok(legend && /paintComponentNote\(/.test(legend[1]),
    'paintLegend no longer refreshes the component caption, so the caption goes ' +
    'stale as soon as the mode or component changes');

  // The DOM node the caption writes into, and the marker the render check reads.
  const html = readFileSync(new URL('../../web/templates/job.html', import.meta.url), 'utf8');
  assert.match(html, /id="component-note"/,
    'the page no longer has the #component-note element the caption and the ' +
    'render check both address');
});
