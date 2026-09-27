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
